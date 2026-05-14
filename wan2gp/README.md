# Wan2GP on Modal

Deploys [Wan2GP](https://github.com/deepbeepmeep/Wan2GP) — AI video generation with a Gradio UI — to [Modal](https://modal.com) with persistent GPU-backed containers.

![WAN2GP result](../docs/images/Screenshot%202026-05-14%20at%2015.58.01.png)

## How It Works

```
┌─────────────────────────────────────────────────────┐
│  Your browser                                       │
│    │                                                 │
│    ▼                                                 │
│  https://xxxx.gradio.live  ◄── primary URL (no proxy)│
│    │                                                 │
│    ▼                                                 │
│  Gradio server (wgp.py subprocess, port 7860)       │
│    │                                                 │
│    ▼                                                 │
│  GPU container (Modal, A100-80GB)                   │
│    │                                                 │
│    ▼                                                 │
│  Modal Volume (persistent: ckpts, loras, outputs)   │
└─────────────────────────────────────────────────────┘
```

**Why Gradio share URL instead of Modal URL?** Modal's web endpoint proxies HTTP traffic through its own reverse proxy, which causes `Content-Length` mismatch errors when Gradio streams responses (SSE, progressive loading, file uploads). Gradio's `--share` flag creates a direct tunnel (`*.gradio.live`) that bypasses the proxy entirely.

The Modal URL (`*.modal.run`) still exists as a fallback but will likely hit those streaming errors. **Always use the Gradio share URL.**

## Prerequisites

1. **Python 3.12+**
2. **Modal CLI** — install and authenticate:
   ```bash
   pip install modal
   modal setup
   ```
3. **Modal account** with GPU quota

## Quick Start

### Development (ephemeral, logs in terminal)

```bash
modal serve wan2gp/wan2gp.py
```

Wait for the Gradio share URL to appear in the logs:
```
GRADIO: Running on public URL: https://xxxx.gradio.live
```

Open that URL in your browser. Press `Ctrl+C` to stop.

### Production (persistent)

```bash
modal deploy wan2gp/wan2gp.py
```

The app stays running with autoscaling. The Gradio share URL changes on each container restart — find the current one in the [Modal dashboard logs](https://modal.com/apps).

## Configuration

All config is at the top of `wan2gp.py` — edit these constants to customize:

| Constant | Default | Description |
|---|---|---|
| `GPU_TYPE` | `A100-80GB` | GPU type. Options: `T4`, `L4`, `A10`, `L40S`, `A100`, `A100-40GB`, `A100-80GB`, `H100`, `H200`, `B200` |
| `TIMEOUT` | `3600` | Max seconds a container stays alive |
| `MAX_CONCURRENT_INPUTS` | `3` | Max concurrent requests per container |
| `WAN2GP_PROFILE` | `"1"` | Offloading profile: `1` = high VRAM, `2.5` = balanced, `3` = low VRAM |
| `GRADIO_SHARE` | `True` | Create a public `*.gradio.live` tunnel URL |
| `min_containers` | `1` | Always keep 1 container warm (so the Gradio URL stays alive) |

## Persistent Storage

A Modal Volume (`wan2gp-data`) persists across container restarts. Model checkpoints, LoRA weights, generated videos, and HuggingFace caches are all stored here — no re-downloading on cold starts.

```
wan2gp-data/
├── ckpts/          # model checkpoints (downloaded automatically)
├── loras/          # LoRA weights
│   ├── ltx2/
│   └── ltx2_22B/
├── outputs/        # generated videos and images
└── cache/          # HF hub, transformers, torch caches
```

### Inspecting the volume

```bash
modal volume ls wan2gp-data
modal volume ls wan2gp-data ckpts/
modal volume get wan2gp-data outputs/my_video.mp4 .
```

## Architecture Details

### Container image

Built once and cached by Modal. Layers:

1. Debian slim + Python 3.12
2. System packages (git, ffmpeg, libgl, etc.)
3. Clone Wan2GP repo
4. Install Python dependencies (requirements.txt, PyTorch, xformers, onnxruntime-gpu)
5. Patch matplotlib backend for headless server

### Container startup lifecycle

1. **`setup()`** (`@modal.enter`) — runs once per container:
   - Logs GPU info (nvidia-smi + torch.cuda)
   - Creates volume directory structure
   - Symlinks repo dirs (`ckpts/`, `loras/`, `outputs/`) into the volume
   - Sets cache env vars (`HF_HOME`, `TORCH_HOME`, etc.) to volume paths
   - Commits volume

2. **`launch()`** (`@modal.web_server`) — starts the Gradio server:
   - Runs `wgp.py` as a subprocess with `--share` and `--listen`
   - Streams subprocess output to structured Modal logs

### Cost implications

With `min_containers=1`, one A100-80GB container is always running. This means:
- **~$3.94/hr** (check [Modal pricing](https://modal.com/pricing) for current rates)
- The Gradio share URL stays alive without needing to trigger the Modal URL
- To save costs, set `min_containers=0` but you'll need to hit the Modal URL first to cold-start the container

## Common Commands

| Command | Purpose |
|---|---|
| `modal serve wan2gp/wan2gp.py` | Dev mode with hot-reload |
| `modal deploy wan2gp/wan2gp.py` | Production deploy |
| `modal app list` | List running apps |
| `modal app stop wan2gp` | Stop the app |
| `modal volume ls wan2gp-data` | Inspect persistent volume |
| `modal volume get wan2gp-data <path> .` | Download a file from volume |

## Troubleshooting

| Issue | Solution |
|---|---|
| Content-Length error in browser | You're using the Modal URL. Switch to the Gradio `*.gradio.live` URL from the logs |
| Container not starting / no Gradio URL | Check Modal dashboard logs for errors. First deploy takes time to build the image (~5 min) |
| Gradio share URL expired | Share URLs last 72 hours. Restart the container with `modal app stop wan2gp` then re-deploy |
| Out of GPU memory | Lower `WAN2GP_PROFILE` to `"2.5"` or `"3"` for more aggressive offloading, or switch to a GPU with more VRAM |
| Want to save costs | Set `min_containers=0` — but container won't start until the Modal URL is hit first |

### Notes 

**DO NOT USE** use the link from modal, wait until the container run and use the gradio link, open it in your browser.
![WAN2GP result](../docs/images/Screenshot%202026-05-14%20at%2015.45.58.png)

This is to avoid isuse the dropdown trigger not work when we changes the model
![WAN2GP result](../docs/images/Screenshot%202026-05-14%20at%2015.18.45.png)