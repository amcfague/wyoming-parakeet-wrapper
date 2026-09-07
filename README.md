# Wyoming Parakeet TDT v2

Wyoming speech-to-text for Home Assistant using NVIDIA Parakeet TDT v2 through ONNX Runtime CUDA.

Requires a Linux x86_64 host with an NVIDIA GPU, a current NVIDIA driver, Docker, and the NVIDIA Container Toolkit.

```sh
docker compose up --build
```

The first start downloads the model into the `parakeet-models` Docker volume. Add Home Assistant's Wyoming Protocol integration with the Docker host and port `10300`.

The model accepts English 16 kHz mono 16-bit PCM. Parakeet TDT is full-utterance ASR, so transcription begins when Home Assistant sends `audio-stop`.

```sh
docker compose run --rm parakeet --max-audio-seconds 30
```
