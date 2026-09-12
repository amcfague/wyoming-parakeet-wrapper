import asyncio
import io
import threading
import unittest
from types import SimpleNamespace

import numpy as np
import soundfile as sf
from aiohttp import FormData
from aiohttp.test_utils import TestClient, TestServer

from http_api import make_app, recognize, words_from_tokens


class FakeModel:
    def __init__(self):
        self.calls = []

    def with_timestamps(self):
        return self

    def recognize(self, waveform, *, sample_rate):
        self.calls.append(len(waveform))
        return SimpleNamespace(text="Hello world", tokens=["\u2581Hel", "lo", "\u2581world"],
                               timestamps=[0.0, 0.08, 0.4])


class HttpTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.model = FakeModel()
        self.client = TestClient(TestServer(make_app(self.model, asyncio.Lock(), "parakeet")))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def upload(self, fmt="WAV", seconds=1, language="en", rate=16000):
        audio = io.BytesIO()
        sf.write(audio, np.zeros(seconds * rate), rate, format=fmt)
        form = FormData()
        form.add_field("file", audio.getvalue(), filename="audio." + fmt.lower())
        form.add_field("model", "whisper-1")
        form.add_field("language", language)
        form.add_field("response_format", "verbose_json")
        form.add_field("timestamp_granularities[]", "word")
        form.add_field("timestamp_granularities[]", "segment")
        return await self.client.post("/v1/audio/transcriptions", data=form)

    async def test_minus_pod_uploads_and_offsets(self):
        for fmt in ("WAV", "FLAC"):
            response = await self.upload(fmt, seconds=42)
            self.assertEqual(response.status, 200, await response.text())
            body = await response.json()
            self.assertEqual(body["duration"], 42)
            self.assertEqual(len(body["segments"]), 3)
            self.assertEqual(body["segments"][0]["words"][0]["word"], "Hello")
            self.assertGreater(body["segments"][1]["start"], 18)
            self.assertTrue(all(w["start"] <= w["end"] <= 42 for w in body["words"]))
        self.assertTrue(all(n <= 20 * 16000 for n in self.model.calls))

    async def test_rejects_bad_inputs(self):
        with self.assertLogs("http_api", level="WARNING") as logs:
            response = await self.client.post("/v1/audio/transcriptions", json={})
        self.assertEqual(response.status, 400)
        self.assertIn("expected multipart/form-data", logs.output[0])
        for kwargs in ({"language": "de"}, {"rate": 8000}, {"seconds": 0}):
            response = await self.upload(**kwargs)
            self.assertEqual(response.status, 400, await response.text())
        self.assertEqual(self.model.calls, [])

    async def test_standalone_word_boundary(self):
        result = SimpleNamespace(text="a test", tokens=["a", "\u2581", "test"], timestamps=[0, 0.1, 0.2])
        self.assertEqual([w["word"] for w in words_from_tokens(result, 0, 1)], ["a", "test"])

    async def test_model_failure(self):
        def fail(*args, **kwargs):
            raise RuntimeError("inference failure")
        self.model.recognize = fail
        response = await self.upload()
        self.assertEqual(response.status, 500)

    async def test_cancel_keeps_model_locked(self):
        started, release = threading.Event(), threading.Event()
        def blocking(*args, **kwargs):
            started.set()
            release.wait(5)
        lock = asyncio.Lock()
        task = asyncio.create_task(recognize(SimpleNamespace(recognize=blocking), lock, np.zeros(1)))
        await asyncio.to_thread(started.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        try:
            self.assertTrue(lock.locked())
        finally:
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertFalse(lock.locked())


if __name__ == "__main__":
    unittest.main()
