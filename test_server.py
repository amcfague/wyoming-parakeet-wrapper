import asyncio
import unittest
from functools import partial

import numpy as np
from wyoming.asr import Transcript, Transcribe
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.error import Error
from wyoming.server import AsyncTcpServer

from server import (
    CHANNELS,
    RATE,
    WIDTH,
    ParakeetHandler,
    make_info,
    pcm_to_waveform,
    supported_language,
)


class FakeModel:
    def __init__(self):
        self.waveform = None
        self.error = None

    def recognize(self, waveform, *, sample_rate):
        if self.error is not None:
            raise self.error
        self.waveform = waveform
        assert sample_rate == RATE
        return "turn on the lights"


class ServerTest(unittest.TestCase):
    def test_pcm_and_language_validation(self):
        np.testing.assert_allclose(
            pcm_to_waveform(b"\x00\x80\x00\x00\xff\x7f"),
            [-1.0, 0.0, 32767 / 32768],
        )
        self.assertTrue(supported_language("en-US"))
        self.assertFalse(supported_language("de-DE"))
        with self.assertRaises(ValueError):
            pcm_to_waveform(b"\x00")


class WyomingFlowTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.model = FakeModel()
        self.server = AsyncTcpServer("127.0.0.1", 0)
        await self.server.start(
            partial(
                ParakeetHandler,
                make_info(),
                self.model,
                asyncio.Lock(),
                RATE * WIDTH * CHANNELS,
            )
        )
        self.port = self.server._server.sockets[0].getsockname()[1]

    async def asyncTearDown(self):
        await self.server.stop()

    async def test_transcribes_audio_stream(self):
        async with AsyncTcpClient("127.0.0.1", self.port) as client:
            await client.write_event(Transcribe(language="en-US").event())
            await client.write_event(AudioStart(RATE, WIDTH, CHANNELS).event())
            await client.write_event(
                AudioChunk(RATE, WIDTH, CHANNELS, b"\x00\x80\x00\x00\xff\x7f").event()
            )
            await client.write_event(AudioStop().event())

            event = await client.read_event()

        self.assertTrue(Transcript.is_type(event.type))
        self.assertEqual(Transcript.from_event(event).text, "turn on the lights")
        np.testing.assert_allclose(self.model.waveform, [-1.0, 0.0, 32767 / 32768])

    async def test_returns_error_when_model_fails(self):
        self.model.error = RuntimeError("CUDA failed")
        async with AsyncTcpClient("127.0.0.1", self.port) as client:
            await client.write_event(AudioStart(RATE, WIDTH, CHANNELS).event())
            await client.write_event(AudioStop().event())
            event = await client.read_event()

        self.assertTrue(Error.is_type(event.type))


if __name__ == "__main__":
    unittest.main()
