# ComfyUI on Modal

Deploy [ComfyUI](https://github.com/comfyanonymous/ComfyUI) on [Modal](https://modal.com) with modular plugin/model management, GPU snapshots, and an optional OpenAI-compatible video generation API.

Pre-configured for the [VideoFlow LTX 2.3 All-in-One v3.0](https://civitai.com/models/1815300) workflow.

## Quick Start

```bash
# 1. Copy the example configs and edit them
cp comfyui/models_example.py comfyui/models.py
cp comfyui/plugins_example.py comfyui/plugins.py

# 2. (Optional) Create a HuggingFace secret for faster model downloads
modal secret create huggingface-secret HF_TOKEN=hf_YOUR_TOKEN

# 3. Deploy
modal deploy comfyui/comfyui.py
```

## Configuration

### Models (`models.py`)

Two categories:

| Key | Type | Source |
|---|---|---|
| `MODELS_HF` | `list[dict]` | HuggingFace Hub (`repo_id` + `filename`) |
| `MODELS_URL` | `list[dict]` | Direct URL via aria2c (`url` + `filename`) |

Each entry has a `model_dir` — relative paths resolve to `/root/comfy/ComfyUI/models/`.

### Plugins (`plugins.py`)

Three install types:

| Key | Type | How |
|---|---|---|
| `PLUGINS_REGISTRY` | `list[str]` | `comfy node install <id>` |
| `PLUGINS_GIT` | `list[dict]` | `git clone` + optional `pip install -r` |
| `PLUGINS_PIP` | `list[str]` | `pip install <pkg>` |

### Video API (`workflows.py`)

Map model IDs to workflow JSONs for the OpenAI-compatible endpoint:

```python
WORKFLOWS = {
    "ltx2.3": {
        "file": "workflows/ltx23_v30.json",
        "param_map": {"prompt": "1879.widgets.0"},
    },
}
```

## Usage

```bash
# Dev mode (ephemeral, logs in terminal)
modal serve comfyui/comfyui.py

# Production (persistent, auto-scaling)
modal deploy comfyui/comfyui.py

# Test video generation
modal run comfyui/comfyui.py
modal run comfyui/comfyui.py --prompt "A cinematic shot of a sunset over the ocean"
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/v1/video/generations` | POST | Submit a video generation task |
| `/v1/video/generations/{id}` | GET | Check task status |
| `/health` | GET | Health check |
| `/*` | * | Proxied to ComfyUI UI / API |

### Example: Video Generation

```bash
curl -X POST "https://your-app--comfyui-proxy.modal.run/v1/video/generations" \
  -H "Content-Type: application/json" \
  -d '{"model": "ltx2.3", "prompt": "A cat walking through a neon-lit Tokyo alley"}'
```

Response:
```json
{
  "id": "abc123...",
  "status": "completed",
  "output": [
    {"filename": "ComfyUI_00001_.mp4", "subfolder": "", "type": "output", "url": "/view?filename=..."}
  ]
}
```

## Architecture

```
Client
  │
  ▼
Proxy (port 8000, exposed) ─────────────────────────────────
  │  /v1/video/*  → Video API handler (inject + submit + poll)
  │  /*            → Reverse proxy to ComfyUI (port 8001)
  │
  ▼
ComfyUI (port 8001, internal)
  │
  ▼
Modal Volume (hf-hub-cache) ← Persistent model storage
Modal GPU Snapshot ← Fast warm starts
```

## Configuration Constants

All tunables at the top of `comfyui.py`:

| Constant | Default | Description |
|---|---|---|
| `GPU_TYPE` | `L40S` | GPU type |
| `SCALEDOWN_WINDOW` | `60` | Idle seconds before shutdown |
| `TIMEOUT` | `3600` | Max container lifetime (seconds) |
| `MAX_CONCURRENT_INPUTS` | `10` | Max concurrent requests per container |
| `PROXY_PORT` | `8000` | Exposed proxy port |
| `COMFY_PORT` | `8001` | Internal ComfyUI port |

## Persistent Storage

| Volume | Mount Path | Purpose |
|---|---|---|
| `hf-hub-cache` | `/cache` | Model weights (HF, URL downloads) |

Models only download once. Subsequent cold starts are much faster.

## GPU Selection Guide

| GPU | VRAM | Best For |
|---|---|---|
| L40S | 48 GB | LTX 2.3 fp8 checkpoint (29 GB) + text encoders |
| A100-80GB | 80 GB | LTX 2.3 bf16 checkpoint (46 GB), Wan 2.2, heavy workflows |
| H100 | 80 GB | Fastest inference, auto-upgraded to H200 |
| H200 | 141 GB | Multiple models, largest workflows |

## Adding a New Workflow

1. Download the workflow JSON file
2. Place it in `comfyui/workflows/`
3. Determine required custom nodes and add them to `plugins.py`
4. Determine required models and add them to `models.py`
5. Add the OpenAI API mapping in `workflows.py`
6. Redeploy: `modal deploy comfyui/comfyui.py`

## Troubleshooting

- **models.py not found**: Run `cp comfyui/models_example.py comfyui/models.py` and edit it
- **plugins.py not found**: Run `cp comfyui/plugins_example.py comfyui/plugins.py` and edit it
- **ComfyUI won't start**: Check GPU VRAM — the fp8 checkpoint needs ~29 GB + overhead
- **Slow model downloads**: Set up a HuggingFace access token: `modal secret create huggingface-secret HF_TOKEN=hf_...`
- **Custom node missing**: Add the node ID to `PLUGINS_REGISTRY` in `plugins.py` and redeploy
