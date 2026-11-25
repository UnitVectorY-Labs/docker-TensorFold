# docker-TensorFold

CUDA-enabled `tensorfold-native` container built from upstream [ashhart/TensorFold](https://github.com/ashhart/TensorFold) for NVIDIA DGX Spark class devices and equivalent GB10-based variants.

This image compiles TensorFold's native Zig/CUDA server directly from source targeting `-Dgpu=spark` (`sm_121` / compute capability 12.1) along with `libtfgrammar.so` (pinned upstream `xgrammar`) for structured output and tool-call grammar enforcement.

## Use Case

Use this container when you want a reproducible `TensorFold` CUDA inference server image for:

- NVIDIA DGX Spark systems (`sm_121`)
- Other GB10-class devices with equivalent CUDA capability
- Serving TensorFold-compatible MLX / NVFP4 safetensors checkpoints with native MTP / DFlash speculative decoding and OpenAI-compatible HTTP APIs

The image defaults to a non-root `tensorfold` user (`uid:gid 1000:1000`) with `HOME=/home/tensorfold` and `HF_HOME=/home/tensorfold/.cache/huggingface`. No model weights are baked into the image.

## Example Run Commands

TensorFold requires checkpoints to be downloaded into the Hugging Face cache before serving:

```bash
docker run --rm \
  --user 1000:1000 \
  -e HOME=/home/tensorfold \
  -e HF_HOME=/home/tensorfold/.cache/huggingface \
  -v "$HOME/.cache/huggingface:/home/tensorfold/.cache/huggingface" \
  ghcr.io/unitvectory-labs/docker-tensorfold:current \
  pull TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP
```

Serve a cached checkpoint on GPU:

```bash
docker run -d --rm \
  --pull=always \
  --gpus all \
  --name tensorfold \
  -p 8080:8080 \
  --user 1000:1000 \
  -e HOME=/home/tensorfold \
  -e HF_HOME=/home/tensorfold/.cache/huggingface \
  -v "$HOME/.cache/huggingface:/home/tensorfold/.cache/huggingface" \
  ghcr.io/unitvectory-labs/docker-tensorfold:current \
  serve TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP \
  --host 0.0.0.0 \
  --port 8080 \
  --name flash-next \
  --parallel 2 \
  --context 204800
```

## Tags

- `ghcr.io/unitvectory-labs/docker-tensorfold:current` — built from the latest upstream TensorFold GitHub release tag (`amd64` + `arm64`, `sm_121`).
- `ghcr.io/unitvectory-labs/docker-tensorfold:master` — built from upstream TensorFold `main` (`amd64` + `arm64`, `sm_121`).

## Non-Root Runtime Notes

If you pass `--user`, keep `HOME`, `XDG_CACHE_HOME`, and `HF_HOME` aligned with directories that user can write to. The image ships with `/home/tensorfold/.cache/huggingface` and `/home/tensorfold/.cache/tensorfold` owned by `1000:1000`.

## License

This repository's Dockerfile and automation are licensed under the MIT License (see [LICENSE](LICENSE)).
Upstream [TensorFold](https://github.com/ashhart/TensorFold) and `xgrammar` are licensed under the Apache License 2.0; upstream license and notice files are included in the image under `/usr/local/share/licenses/tensorfold/`.
