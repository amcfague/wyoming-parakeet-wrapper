# Wyoming Parakeet TDT v2

Wyoming speech-to-text for Home Assistant using NVIDIA Parakeet TDT v2 through ONNX Runtime CUDA.

Requires a Linux x86_64 host with an NVIDIA GPU, a current NVIDIA driver, Docker, and the NVIDIA Container Toolkit.

```sh
docker compose up -d
```

The first start downloads the model into the `parakeet-models` Docker volume. Add Home Assistant's Wyoming Protocol integration with the Docker host and port `10300`.

The model accepts English 16 kHz mono 16-bit PCM. Parakeet TDT is full-utterance ASR, so transcription begins when Home Assistant sends `audio-stop`.

```sh
docker compose run --rm parakeet --max-audio-seconds 30
```

## MinusPod / OpenAI-Compatible HTTP

The same model also serves `POST /v1/audio/transcriptions` on port `8000`.
Configure MinusPod with:

```sh
WHISPER_BACKEND=openai-api
WHISPER_API_BASE_URL=http://YOUR_DOCKER_HOST:8000/v1
WHISPER_API_MODEL=nemo-parakeet-tdt-0.6b-v2
WHISPER_LANGUAGE=en
WHISPER_DEVICE=cpu
```

Leave the API key empty. This service has no authentication; use it on a trusted network.
The `whisper-1` model name is accepted as an alias for Parakeet v2.

Uploads must be 16 kHz mono WAV or FLAC, which MinusPod produces. The limit is
256 MiB per upload and four hours of decoded audio. Responses support `json`,
`text`, and `verbose_json`. Verbose responses include segments, nested words,
and a top-level word list. Word starts come from Parakeet token emissions;
word ends are estimates, not forced alignments. Check ad-boundary accuracy on
your episodes before relying on automatic edits.

Long uploads are processed in segments of at most 20 seconds, cutting near
quiet audio where possible. A cut can still split a word in continuous speech.
The model lock is released between segments so Home Assistant can transcribe
while a podcast is in progress. Up to four HTTP uploads are accepted concurrently,
matching MinusPod's four workers. Inference remains serialized between segments.
Additional uploads receive HTTP 429 with `Retry-After`.

Use `--http-host` and `--http-port` to change the HTTP listener. For a local build:

```sh
docker build --platform linux/amd64 -t parakeet-http .
docker run --rm --gpus all -p 10300:10300 -p 8000:8000 -v parakeet-models:/models parakeet-http
```

Protocol tests use a fake model and require no GPU:

```sh
python -m unittest -v test_server test_http_api
```
