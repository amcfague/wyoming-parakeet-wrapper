"""MinusPod-compatible transcription using the shared Parakeet instance."""

import asyncio
import logging
import tempfile

import numpy as np
import soundfile as sf
from aiohttp import web

RATE = 16_000
MAX_UPLOAD = 256 * 1024 * 1024
LOGGER = logging.getLogger(__name__)


@web.middleware
async def log_rejections(request, handler):
    try:
        return await handler(request)
    except web.HTTPException as error:
        if error.status >= 400:
            LOGGER.warning("HTTP %s %s rejected (%s): %s", request.method, request.path,
                           error.status, error.text)
        raise


async def recognize(model, lock, waveform):
    async with lock:
        task = asyncio.create_task(asyncio.to_thread(model.recognize, waveform, sample_rate=RATE))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # A cancelled request must not release the model while its thread is running.
            await task
            raise


def words_from_tokens(result, offset, duration):
    if not result.text.strip():
        return []
    if result.tokens is None or result.timestamps is None:
        raise ValueError("model did not return token timestamps")
    words = []
    new_word = False
    for token, timestamp in zip(result.tokens, result.timestamps, strict=True):
        start = offset + min(duration, max(0.0, float(timestamp)))
        new_word = new_word or token.startswith("\u2581") or token.startswith(" ")
        token = token.replace("\u2581", " ")
        if new_word or not words:
            if token.strip():
                words.append({"word": token.strip(), "start": start, "end": start})
                new_word = False
        else:
            words[-1]["word"] += token
            words[-1]["end"] = start
    for i, word in enumerate(words):
        # Token emissions are not alignments: cap estimated ends to avoid spanning silence.
        next_start = words[i + 1]["start"] if i + 1 < len(words) else offset + duration
        word["end"] = min(next_start, word["end"] + 0.16, offset + duration)
    return words


def read_segment(audio):
    start = audio.tell()
    waveform = audio.read(20 * RATE, dtype="float32")
    if len(waveform) == 20 * RATE and audio.tell() < len(audio):
        # Cut at the quietest 20 ms frame in the final two seconds.
        tail = waveform[-2 * RATE:].reshape(-1, 320)
        cut = len(waveform) - 2 * RATE + int(np.argmin(np.mean(tail * tail, axis=1))) * 320 + 160
        waveform = waveform[:cut]
        audio.seek(start + cut)
    return start / RATE, waveform


def make_app(model, lock, model_name):
    timestamped = model.with_timestamps()
    active_uploads = 0

    async def transcribe(request):
        nonlocal active_uploads
        if request.content_type != "multipart/form-data":
            raise web.HTTPBadRequest(text="expected multipart/form-data")
        if active_uploads >= 4:
            raise web.HTTPTooManyRequests(text="four uploads are already being transcribed", headers={"Retry-After": "2"})
        active_uploads += 1
        try:
            with tempfile.TemporaryFile() as upload:
                fields = {}
                size = 0
                found_file = False
                part_count = 0
                reader = await request.multipart()
                async with asyncio.timeout(300):
                    async for part in reader:
                        part_count += 1
                        if part_count > 32:
                            raise ValueError("too many form parts")
                        if part.name == "file":
                            if found_file:
                                raise ValueError("only one audio file is supported")
                            found_file = True
                            while chunk := await part.read_chunk():
                                size += len(chunk)
                                if size > MAX_UPLOAD:
                                    raise web.HTTPRequestEntityTooLarge(max_size=MAX_UPLOAD, actual_size=size)
                                await asyncio.to_thread(upload.write, chunk)
                        else:
                            value = bytearray()
                            while chunk := await part.read_chunk():
                                size += len(chunk)
                                value.extend(chunk)
                                if size > MAX_UPLOAD or len(value) > 4096:
                                    raise ValueError("form fields too large")
                            fields[part.name] = value.decode("utf-8")
                if not found_file:
                    raise ValueError("file is required")
                if fields.get("language", "en").lower() not in {"en", "en-us", "auto", ""}:
                    raise ValueError("Parakeet v2 supports English only")
                if fields.get("model", model_name) not in {model_name, "whisper-1"}:
                    raise ValueError(f"use model {model_name} or whisper-1")
                response_format = fields.get("response_format", "json")
                if response_format not in {"json", "verbose_json", "text"}:
                    raise ValueError("supported response formats: json, verbose_json, text")
                upload.seek(0)
                with sf.SoundFile(upload) as audio:
                    if audio.format not in {"WAV", "WAVEX", "FLAC"} or audio.samplerate != RATE or audio.channels != 1:
                        raise ValueError("expected 16 kHz mono WAV or FLAC (as sent by MinusPod)")
                    duration = len(audio) / RATE
                    if duration > 4 * 60 * 60 or not duration:
                        raise ValueError("audio must be nonempty and at most four hours")
                    segments = []
                    while audio.tell() < len(audio):
                        offset, waveform = await asyncio.to_thread(read_segment, audio)
                        result = await recognize(timestamped, lock, waveform)
                        words = words_from_tokens(result, offset, len(waveform) / RATE)
                        if words:
                            segments.append({"id": len(segments), "start": words[0]["start"],
                                             "end": words[-1]["end"], "text": result.text, "words": words})
                text = " ".join(segment["text"].strip() for segment in segments)
                if response_format == "text":
                    return web.Response(text=text)
                body = {"text": text}
                if response_format == "verbose_json":
                    body.update(task="transcribe", language="english", duration=duration,
                                segments=segments, words=[w for s in segments for w in s["words"]])
                return web.json_response(body)
        except web.HTTPException:
            raise
        except (ValueError, sf.LibsndfileError) as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        except TimeoutError as error:
            raise web.HTTPRequestTimeout(text="upload timed out") from error
        except Exception as error:
            logging.exception("HTTP transcription failed")
            raise web.HTTPInternalServerError(text="transcription failed") from error
        finally:
            active_uploads -= 1

    app = web.Application(client_max_size=MAX_UPLOAD, middlewares=[log_rejections])
    app.router.add_post("/v1/audio/transcriptions", transcribe)
    return app
