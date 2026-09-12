"""Wyoming ASR server for NVIDIA Parakeet TDT v2."""

import argparse
import asyncio
import logging
from functools import partial

import numpy as np
from wyoming.asr import Transcript, Transcribe
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

MODEL = "nemo-parakeet-tdt-0.6b-v2"
RATE = 16_000
WIDTH = 2
CHANNELS = 1
LOGGER = logging.getLogger(__name__)


def supported_language(language: str | None) -> bool:
    return language is None or language.lower() in {"en", "en-us"}


def pcm_to_waveform(audio: bytes) -> np.ndarray:
    if len(audio) % WIDTH:
        raise ValueError("audio payload must contain complete 16-bit samples")

    return np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768.0


def load_model():
    import ctypes
    from pathlib import Path

    import onnx_asr
    import onnxruntime as ort

    ort.preload_dlls(directory="")
    # Fail before ORT can silently fall back to CPU when CUDA libraries are missing.
    ctypes.CDLL(str(Path(ort.__file__).parent / "capi" / "libonnxruntime_providers_cuda.so"))

    model = onnx_asr.load_model(
        MODEL,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    model.recognize(np.zeros(RATE, dtype=np.float32), sample_rate=RATE)
    return model


async def recognize(model, lock, waveform):
    async with lock:
        task = asyncio.create_task(asyncio.to_thread(model.recognize, waveform, sample_rate=RATE))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # Keep the model locked until its worker thread finishes.
            await task
            raise


class ParakeetHandler(AsyncEventHandler):
    def __init__(
        self,
        info: Info,
        model,
        model_lock: asyncio.Lock,
        max_audio_bytes: int,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        super().__init__(reader, writer)
        self.info = info
        self.model = model
        self.model_lock = model_lock
        self.max_audio_bytes = max_audio_bytes
        self.audio = bytearray()
        self.audio_started = False
        self.language: str | None = None

    async def fail(self, text: str) -> bool:
        await self.write_event(Error(text=text).event())
        self.audio.clear()
        self.audio_started = False
        return False

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.info.event())
            return True

        if Transcribe.is_type(event.type):
            self.language = Transcribe.from_event(event).language
            if not supported_language(self.language):
                return await self.fail("Parakeet TDT v2 supports English only")
            return True

        if AudioStart.is_type(event.type):
            audio_start = AudioStart.from_event(event)
            if (audio_start.rate, audio_start.width, audio_start.channels) != (
                RATE,
                WIDTH,
                CHANNELS,
            ):
                return await self.fail("expected 16 kHz, mono, 16-bit PCM audio")
            self.audio.clear()
            self.audio_started = True
            return True

        if AudioChunk.is_type(event.type):
            if not self.audio_started:
                return await self.fail("audio-chunk received before audio-start")

            chunk = AudioChunk.from_event(event)
            if (chunk.rate, chunk.width, chunk.channels) != (RATE, WIDTH, CHANNELS):
                return await self.fail("audio format changed during transcription")
            if len(self.audio) + len(chunk.audio) > self.max_audio_bytes:
                return await self.fail("audio exceeds configured duration limit")

            self.audio.extend(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            if not self.audio_started:
                return await self.fail("audio-stop received before audio-start")

            try:
                waveform = pcm_to_waveform(bytes(self.audio))
            except ValueError as err:
                return await self.fail(str(err))
            finally:
                self.audio.clear()
                self.audio_started = False

            # ponytail: one local model instance; add a model pool only if concurrent requests matter.
            try:
                text = await recognize(self.model, self.model_lock, waveform)
            except Exception:
                LOGGER.exception("Transcription failed")
                return await self.fail("transcription failed")

            await self.write_event(Transcript(text=text).event())
            return True

        return True

    async def disconnect(self) -> None:
        self.audio.clear()
        self.audio_started = False


def make_info() -> Info:
    return Info(
        asr=[
            AsrProgram(
                name="Parakeet TDT v2",
                description="NVIDIA Parakeet TDT v2 via ONNX Runtime",
                attribution=Attribution(
                    name="NVIDIA",
                    url="https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2",
                ),
                installed=True,
                version="0.6b-v2",
                models=[
                    AsrModel(
                        name=MODEL,
                        description="English speech-to-text",
                        attribution=Attribution(
                            name="NVIDIA",
                            url="https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2",
                        ),
                        installed=True,
                        languages=["en"],
                        version="0.6b-v2",
                    )
                ],
            )
        ]
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="tcp://0.0.0.0:10300")
    parser.add_argument("--max-audio-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.max_audio_seconds < 1:
        parser.error("--max-audio-seconds must be positive")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOGGER.info("Loading %s", MODEL)
    model = await asyncio.to_thread(load_model)
    LOGGER.info("Model ready")

    server = AsyncServer.from_uri(args.uri)
    await server.run(
        partial(
            ParakeetHandler,
            make_info(),
            model,
            asyncio.Lock(),
            args.max_audio_seconds * RATE * WIDTH * CHANNELS,
        )
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
