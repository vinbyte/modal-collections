# Modal Collections

A monorepo of GPU workloads deployed on [Modal](https://modal.com). Each subfolder is a self-contained app you can deploy independently.

## Structure

```
modal-collections/
├── wan2gp/          # Wan2GP video generation (Gradio)
│   ├── wan2gp.py
│   └── .venv/       # (optional, for local IDE support)
├── vllm/            # vLLM LLM inference (OpenAI-compatible API)
│   ├── serve.py
│   └── README.md
├── comfyui/         # ComfyUI with modular plugins + video API
│   ├── comfyui.py
│   ├── models_example.py
│   ├── plugins_example.py
│   ├── workflows.py
│   └── workflows/
└── README.md
```

Every app folder contains its own `*.py` entrypoint and optionally a `.venv` for local development. There is no shared state between apps.

## Prerequisites

1. **Python 3.12+**
2. **Modal CLI** — install and authenticate:
   ```bash
   pip install modal
   modal setup
   ```
3. **Modal account** with GPU quota (A100 or better recommended for Wan2GP)

> **No local dependency install needed.** All dependencies (PyTorch, xformers, etc.) are defined in `modal.Image` and installed in the remote container at build time. You don't need to `pip install` or `uv sync` anything before deploying.

## Local Development (Optional)

If you want IDE autocomplete, type checking, or linting locally, you can create a `.venv` per app folder. This is **not required** for deploy — it's purely for editor support.

```bash
cd wan2gp
python -m venv .venv
source .venv/bin/activate
pip install modal         # at minimum, for the modal SDK
# Add any other packages you want autocomplete for, e.g.:
# pip install torch torchvision gradio
```

Each app folder keeps its own `.venv` so they stay isolated. `.venv` directories are gitignored.

## Running an App

> **First time?** Run `modal setup` to authenticate before any `modal` command. You only need to do this once.

### Development (ephemeral)

```bash
modal serve wan2gp/wan2gp.py
```

This spins up the app temporarily. Logs stream to your terminal. Press `Ctrl+C` to stop.

### Production (persistent)

```bash
modal deploy wan2gp/wan2gp.py
```

The app stays running and auto-scales. You get a stable URL for any web endpoints.

### Common commands

| Command | Purpose |
|---|---|
| `modal serve <path>` | Run with hot-reload, logs in terminal |
| `modal deploy <path>` | Deploy persistently with autoscaling |
| `modal app list` | List running apps |
| `modal app stop <name>` | Stop a deployed app |
| `modal volume ls wan2gp-data` | Inspect persistent volume contents |

## App: ComfyUI

ComfyUI with modular plugin/model management, GPU snapshots, and an OpenAI-compatible video generation API. Pre-configured for the VideoFlow LTX 2.3 All-in-One v3.0 workflow.

- **Image**: Debian slim + comfy-cli + starlette proxy
- **GPU**: L40S (48 GB VRAM)
- **Model**: LTX 2.3 fp8 checkpoint (~29 GB) + text encoders + VAEs
- **Storage**: `hf-hub-cache` Modal Volume (models persist across restarts)
- **Endpoint**: ComfyUI UI proxied on port 8000, video API at `/v1/video/generations`

### Quick start

```bash
# Copy configs
cp comfyui/models_example.py comfyui/models.py
cp comfyui/plugins_example.py comfyui/plugins.py

# Deploy
modal deploy comfyui/comfyui.py

# Test
modal run comfyui/comfyui.py --prompt "A cinematic sunset time-lapse"
```

See [`comfyui/README.md`](comfyui/README.md) for full configuration details.

## App: Wan2GP

Wan2GP video generation with a Gradio UI, running on an A100 80GB GPU.

- **Image**: Debian slim + PyTorch 2.8.0 (CUDA 12.8) + xformers
- **GPU**: A100 (profile 1)
- **Storage**: `wan2gp-data` Modal Volume (checkpoints, LoRAs, outputs, cache persist across restarts)
- **Endpoint**: Gradio web server on port 7860

### Volume layout

```
wan2gp-data/
├── ckpts/          # model checkpoints
├── loras/          # LoRA weights
│   ├── ltx2/
│   └── ltx2_22B/
├── outputs/        # generated videos
└── cache/          # HF hub, transformers, torch caches
```

### Deploy

```bash
modal deploy wan2gp/wan2gp.py
```

Modal will print the Gradio URL after the container starts.

## App: vLLM

OpenAI-compatible LLM inference server using vLLM, currently running Qwen3.6-27B AWQ INT4 (4-bit quantized, ~21 GB VRAM) with vision + reasoning support on an A100 80GB GPU at full 262K context length.

- **Image**: NVIDIA CUDA 12.9 + vLLM 0.19.0 + transformers 5.5.0
- **GPU**: A100-80GB (1x)
- **Model**: cyankiwi/Qwen3.6-27B-AWQ-INT4 (AWQ 4-bit, ~21 GB VRAM, full 262K context)
- **Storage**: `huggingface-cache` + `vllm-cache` Modal Volumes (model weights and JIT cache persist across restarts)
- **Endpoint**: OpenAI-compatible API on port 8000 (`/v1/chat/completions`, `/v1/completions`, `/v1/models`)

### Quick test

```bash
modal run vllm/serve.py
modal run vllm/serve.py --content "What is the meaning of life?"
```

### Deploy

```bash
modal deploy vllm/serve.py
```

See [`vllm/README.md`](vllm/README.md) for full API usage examples and configuration details.
