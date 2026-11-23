# MiniMax H3 on Modal

Deploy [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) — an open-weight multimodal video+audio generation model — on Modal using vLLM-Omni.

Generates MP4 files with H.264 video at 24 FPS and native stereo AAC audio from text prompts, images, video clips, and audio references.

## Quick Start

### 1. Get HuggingFace access

Request access on the [model card](https://huggingface.co/MiniMaxAI/MiniMax-H3) (the repo is gated).

### 2. Create Modal secret

```bash
modal secret create huggingface-secret HF_TOKEN=<your-huggingface-token>
```

### 3. Download model (CPU-only, no GPU cost)

```bash
modal run minimax-h3/serve.py::download_model
```

Downloads ~130 GB to the `minimax-h3-models` volume. Takes ~30 minutes first time, instant on subsequent runs.

### 4. Deploy

```bash
# Dev mode (logs in terminal, hot-reloads on save)
modal serve minimax-h3/serve.py

# Production (persistent URL, keep 1 GPU container warm)
MINIMAX_MIN_CONTAINERS=1 modal deploy minimax-h3/serve.py
```

### 5. Generate a video

```bash
export BASE_URL="https://your-app--minimax-h3-minimaxh3-serve.modal.run"

# Text-to-video+audio (2K 9:16 vertical)
curl -X POST "$BASE_URL/v1/videos/sync" \
  -F "prompt=At night, three cats march into a bedroom playing tiny brass instruments, then abruptly file out, with synchronized room ambience." \
  -F "width=1440" \
  -F "height=2560" \
  -F "fps=24" \
  -F "num_inference_steps=50" \
  -F "flow_shift=12" \
  -F "seed=1101" \
  -F 'extra_params={"task":"t2va","duration":5.0,"audio_flow_shift":3.0}' \
  -o output.mp4
```

Or test locally:

```bash
modal run minimax-h3/serve.py \
  --prompt "A golden retriever surfing on a rainbow wave" \
  --duration 8.0
```

## curl Examples

All examples assume the base URL is exported:

```bash
export BASE_URL="https://your-app--minimax-h3-minimaxh3-serve.modal.run"
```

### Text-to-Video+Audio (T2VA) — FL2VA partition

```bash
# 16:9 landscape 2K
curl -X POST "$BASE_URL/v1/videos/sync" \
  -F "prompt=A majestic eagle soaring over a misty mountain range at sunrise, wings catching golden light." \
  -F "width=2560" \
  -F "height=1440" \
  -F "fps=24" \
  -F "num_inference_steps=50" \
  -F "flow_shift=12" \
  -F "seed=42" \
  -F 'extra_params={"task":"t2va","duration":5.0,"audio_flow_shift":3.0}' \
  -o t2va.mp4

# 9:16 vertical 2K
curl -X POST "$BASE_URL/v1/videos/sync" \
  -F "prompt=A ballerina dancing in an empty theater, spotlight following her movements across the stage." \
  -F "width=1440" \
  -F "height=2560" \
  -F "fps=24" \
  -F "num_inference_steps=50" \
  -F "flow_shift=12" \
  -F "seed=99" \
  -F 'extra_params={"task":"t2va","duration":5.0,"audio_flow_shift":3.0}' \
  -o t2va-vertical.mp4

# 15-second max duration, short edge 768px (faster, lower VRAM)
curl -X POST "$BASE_URL/v1/videos/sync" \
  -F "prompt=A time-lapse of a flower blooming from bud to full blossom in a sunlit garden." \
  -F "width=1344" \
  -F "height=768" \
  -F "fps=24" \
  -F "num_inference_steps=50" \
  -F "flow_shift=12" \
  -F "seed=123" \
  -F 'extra_params={"task":"t2va","duration":15.0,"audio_flow_shift":3.0}' \
  -o t2va-15s.mp4
```

### First/Last Frame (FL2VA) — FL2VA partition

Upload 1 image as start frame. Width/height are optional — omit both to inherit the image's aspect ratio at 768px short edge.

```bash
# Single start frame (auto-resolution from image)
curl -X POST "$BASE_URL/v1/videos/sync" \
  -F "prompt=Smooth camera push-in on this scene, gentle ambient sound." \
  -F "fps=24" \
  -F "num_inference_steps=50" \
  -F "flow_shift=12" \
  -F "seed=55" \
  -F "input_reference=@/path/to/start_frame.jpg;type=image/jpeg" \
  -F 'extra_params={"task":"fl2va","duration":5.0,"audio_flow_shift":3.0}' \
  -o fl2va.mp4

# With explicit 2K vertical resolution
curl -X POST "$BASE_URL/v1/videos/sync" \
  -F "prompt=Cinematic reveal of this landscape, clouds drifting, wind through trees." \
  -F "width=1440" \
  -F "height=2560" \
  -F "fps=24" \
  -F "num_inference_steps=50" \
  -F "flow_shift=12" \
  -F "seed=88" \
  -F "input_reference=@/path/to/frame.png;type=image/png" \
  -F 'extra_params={"task":"fl2va","duration":8.0,"audio_flow_shift":3.0}' \
  -o fl2va-2k.mp4
```

### Omni Reference (Ref2VA) — Ref2VA partition

> Requires switching to the Ref2VA partition:
> ```bash
> MINIMAX_PARTITION=Ref2VA modal deploy minimax-h3/serve.py
> ```
>
> **Serving limits** (current vLLM-Omni): exactly 1 image + 1 audio, OR 1+ videos (uses source soundtracks, no separate audio).
> Audio must always be paired with at least one image or video.

#### Image + audio reference

Upload one image via `input_reference` and pass audio as a URL via the `audio_reference` JSON field:

```bash
curl -X POST "$BASE_URL/v1/videos/sync" \
  -F "prompt=A white cat with black mustache markings sits on a couch, lip-syncing precisely to the audio before shifting from confusion to deadpan." \
  -F "width=1344" \
  -F "height=768" \
  -F "fps=24" \
  -F "num_inference_steps=50" \
  -F "flow_shift=12" \
  -F "seed=33" \
  -F "input_reference=@/path/to/cat_photo.jpg;type=image/jpeg" \
  -F 'audio_reference={"audio_url":"http://your-server.com/audio.mp3"}' \
  -F 'extra_params={"task":"ref2va","duration":15.0,"audio_flow_shift":3.0}' \
  -o ref2va-image-audio.mp4
```

> **Note:** `duration` should cover the full audio length. If shorter, the audio is truncated. The audio reference must be accessible via HTTP(S) URL — direct file upload is not supported for audio in this path.

#### Video reference (preserves source audio)

Upload one or more videos via `input_references` (plural, repeated field). Each video's own soundtrack is used — do NOT add `audio_reference`:

```bash
# Single video reference
curl -X POST "$BASE_URL/v1/videos/sync" \
  -F "prompt=Extend this scene with the same motion style and cinematic atmosphere." \
  -F "width=1344" \
  -F "height=768" \
  -F "fps=24" \
  -F "num_inference_steps=50" \
  -F "flow_shift=12" \
  -F "seed=44" \
  -F "input_references=@/path/to/reference.mp4;type=video/mp4" \
  -F 'extra_params={"task":"ref2va","duration":8.0,"audio_flow_shift":3.0}' \
  -o ref2va-video.mp4

# Two video references (e.g., green-screen removal + background replacement)
curl -X POST "$BASE_URL/v1/videos/sync" \
  -F "prompt=Remove the green screen background of Video 1 and replace it with the environment from Video 2. Match the background motion to the character actions." \
  -F "width=1344" \
  -F "height=768" \
  -F "fps=24" \
  -F "num_inference_steps=50" \
  -F "flow_shift=12" \
  -F "seed=66" \
  -F "input_references=@/path/to/subject.mp4;type=video/mp4" \
  -F "input_references=@/path/to/background.mp4;type=video/mp4" \
  -F 'extra_params={"task":"ref2va","duration":15.0,"audio_flow_shift":3.0}' \
  -o ref2va-two-videos.mp4
```

> Uploaded video references are stored only for the request lifetime and deleted after generation.

#### Reference file limits

| Type | Max count | Max duration (total) | Formats | Per-file max |
|---|---|---|---|---|
| Image | 1* (serving path) / 9 (model) | — | JPG, JPEG, PNG, WEBP, HEIC, HEIF | 30 MB |
| Video | 3 | 15s (each 2–15s) | H.264/AVC, H.265/HEVC (AAC/MP3 audio) | 50 MB |
| Audio | 1 URL* (serving path) / 3 (model) | 15s | WAV, MP3 (via HTTP URL only) | 15 MB |
| Request body | — | — | — | 64 MB total |

\* Current vLLM-Omni serving path limit. The model supports more (9 images, 3 audio).

### Async Endpoint (`/v1/videos`)

For long generations, use the async job-polling pattern:

```bash
# Step 1: Submit async job
JOB=$(curl -s -X POST "$BASE_URL/v1/videos" \
  -F "prompt=A time-lapse of a flower blooming from bud to full blossom in a sunlit garden." \
  -F "width=1440" \
  -F "height=2560" \
  -F "fps=24" \
  -F "num_inference_steps=50" \
  -F "flow_shift=12" \
  -F "seed=42" \
  -F 'extra_params={"task":"t2va","duration":5.0,"audio_flow_shift":3.0}')

JOB_ID=$(echo "$JOB" | jq -r '.id')
echo "Job submitted: $JOB_ID"

# Step 2: Poll until complete
while true; do
  RESPONSE=$(curl -s "$BASE_URL/v1/videos/$JOB_ID")
  STATUS=$(echo "$RESPONSE" | jq -r '.status')
  PROGRESS=$(echo "$RESPONSE" | jq -r '.progress // 0')
  echo "Status: $STATUS ($PROGRESS%)"
  if [ "$STATUS" = "completed" ]; then
    curl -s "$BASE_URL/v1/videos/$JOB_ID/content" -o async-output.mp4
    echo "Saved to async-output.mp4"
    break
  elif [ "$STATUS" = "failed" ]; then
    echo "Job failed: $RESPONSE"
    break
  fi
  sleep 10
done
```

### Python client example

```python
import requests
import json
import time

BASE_URL = "https://your-app--minimax-h3-minimaxh3-serve.modal.run"

# Sync generation (blocking)
resp = requests.post(
    f"{BASE_URL}/v1/videos/sync",
    files={
        "prompt": (None, "A golden retriever surfing on a rainbow wave at sunset."),
        "width": (None, "1440"),
        "height": (None, "2560"),
        "fps": (None, "24"),
        "num_inference_steps": (None, "50"),
        "flow_shift": (None, "12"),
        "seed": (None, "1101"),
        "extra_params": (None, json.dumps({
            "task": "t2va",
            "duration": 5.0,
            "audio_flow_shift": 3.0,
        })),
    },
    timeout=1800,
)
with open("output.mp4", "wb") as f:
    f.write(resp.content)
print(f"Saved: {len(resp.content) / 1e6:.1f} MB")

# Async generation (non-blocking, with polling)
resp = requests.post(
    f"{BASE_URL}/v1/videos",
    files={
        "prompt": (None, "A drone shot flying over a lush tropical island at golden hour."),
        "width": (None, "1440"),
        "height": (None, "2560"),
        "fps": (None, "24"),
        "num_inference_steps": (None, "50"),
        "flow_shift": (None, "12"),
        "seed": (None, "777"),
        "extra_params": (None, json.dumps({
            "task": "t2va",
            "duration": 8.0,
            "audio_flow_shift": 3.0,
        })),
    },
    timeout=30,
)
job = resp.json()
job_id = job["id"]
print(f"Job submitted: {job_id}")

while True:
    status = requests.get(f"{BASE_URL}/v1/videos/{job_id}").json()
    print(f"Status: {status['status']}")
    if status["status"] == "completed":
        content = requests.get(f"{BASE_URL}/v1/videos/{job_id}/content").content
        with open("async-output.mp4", "wb") as f:
            f.write(content)
        print(f"Saved: async-output.mp4 ({len(content) / 1e6:.1f} MB)")
        break
    elif status["status"] == "failed":
        print(f"Failed: {status}")
        break
    time.sleep(10)
```

### Health check

```bash
curl -s "$BASE_URL/health"
# => {"status": "healthy"}
```

## Configuration

All settings are configurable via environment variables:

| Variable | Default | Description |
|---|---|---|
| `MINIMAX_GPU_TYPE` | `H100` | GPU type: `H100`, `H100!`, `A100-80GB`, `H200`, `B200` |
| `MINIMAX_N_GPU` | `4` | GPU count: `1`, `2`, `4`, or `8` |
| `MINIMAX_PARTITION` | `FL2VA` | `FL2VA` (t2va, fl2va) or `Ref2VA` (ref2va) |
| `MINIMAX_WIDTH` | `1440` | Output width (multiple of 32) |
| `MINIMAX_HEIGHT` | `2560` | Output height (multiple of 32) |
| `MINIMAX_MIN_CONTAINERS` | `0` | Min warm containers (0 = scale-to-zero, 1 = always warm) |
| `MINIMAX_DLO_RESIDENT_LAYERS` | `20` | Resident layers for 2-GPU DLO mode |

Examples:

```bash
# 2-GPU config with DLO
MINIMAX_N_GPU=2 modal deploy minimax-h3/serve.py

# Ref2VA partition (image/video/audio reference mode)
MINIMAX_PARTITION=Ref2VA modal deploy minimax-h3/serve.py

# 16:9 landscape 2K
MINIMAX_WIDTH=2560 MINIMAX_HEIGHT=1440 modal deploy minimax-h3/serve.py

# Production with always-warm container
MINIMAX_MIN_CONTAINERS=1 modal deploy minimax-h3/serve.py
```

## GPU Configurations

| GPUs | Strategy | Peak VRAM | Host RAM | Notes |
|---|---|---|---|---|---|
| 1× H100 | CPU offload | ~80 GB | 384 GiB (forced) | Accuracy reference path, slow |
| 2× H100 | DLO + TP2 | ~80 GB each | 384 GiB (forced) | Streams weights via PCIe |
| 4× H100 | TP4 | ~30 GB each | Default | Balanced cost/perf (default) |
| 8× H100 | TP8 | ~18 GB each | Default | Fastest |

## API Reference

### `POST /v1/videos/sync`

Synchronous video generation. Returns the MP4 in the response body.

**Form fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | (required) | Text prompt (up to 7,000 chars) |
| `width` | int | `1440` | Output width (multiple of 32) |
| `height` | int | `2560` | Output height (multiple of 32) |
| `fps` | int | `24` | Output FPS (fixed at 24) |
| `num_inference_steps` | int | `50` | Denoising steps |
| `flow_shift` | float | `12` | Video sigma shift |
| `seed` | int | — | Fixed seed for deterministic output |
| `input_reference` | file | — | First frame image (FL2VA) or single image + audio reference (Ref2VA) |
| `input_references` | file(s) | — | One or more video files (Ref2VA multi-video), repeated field |
| `audio_reference` | JSON | — | `{"audio_url":"http://..."}` — audio URL (Ref2VA image+audio) |
| `extra_params` | JSON string | — | Task-specific params (see below) |

**`extra_params` keys:**

| Key | Type | Description |
|---|---|---|
| `task` | string | `t2va`, `fl2va`, or `ref2va` — must match served partition |
| `duration` | float | 4–15 seconds |
| `audio_flow_shift` | float | Audio sigma shift (recommended: `3.0`) |

### `POST /v1/videos`

Async video generation. Returns `{"id": "<job-id>"}` immediately. Poll `GET /v1/videos/{id}` for status (`pending` → `completed`/`failed`). Download result from `GET /v1/videos/{id}/content`.

### `GET /health`

Returns `{"status": "healthy"}`.

## Supported Resolutions

Short edge 768 px or 1440 px, aspect ratios: 21:9, 16:9, 4:3, 1:1, 3:4, 9:16.

| Aspect | 768p | 1440p (2K) |
|---|---|---|
| 16:9 | 1344×768 | 2560×1440 |
| 4:3 | 1024×768 | 1920×1440 |
| 1:1 | 768×768 | 1440×1440 |
| 9:16 | 768×1344 | 1440×2560 |
| 21:9 | 1536×672 | 2976×1248 |

## Architecture

```
┌──────────────────────────────────────────────────┐
│ Modal Container (H100 × N)                       │
│                                                  │
│  vllm serve --omni /mnt/models/MiniMax-H3/FL2VA  │
│  ├── Text/Vision Encoder (Qwen3-VL, 51.5 GB)     │
│  ├── Joint Video/Audio DiT (66.3 GB)             │
│  ├── Video VAE (10 GB)                           │
│  └── Audio VAE (0.6 GB)                          │
│                                                  │
│  Volume: minimax-h3-models → /mnt/models          │
│  ├── .hf-cache/         # HF download cache       │
│  └── MiniMax-H3/        # model weights           │
│      ├── FL2VA/         # t2va + fl2va partition  │
│      └── Ref2VA/        # ref2va partition        │
└──────────────────────────────────────────────────┘
```

Output is an MP4 with H.264 video (24 FPS) + stereo AAC audio (32 kHz), returned synchronously via `POST /v1/videos/sync`.

## Troubleshooting

**"Model not ready" error** — Run the download first:
```bash
modal run minimax-h3/serve.py::download_model
```

**Download fails** — Verify HF_TOKEN is set in the `huggingface-secret` Modal secret and you have access to the model on HuggingFace.

**"Partition not found after download"** — Delete the volume marker and retry:
```bash
modal volume rm minimax-h3-models /MiniMax-H3/.download_complete
modal run minimax-h3/serve.py::download_model
```

**Out of memory** — Try fewer GPUs with offload (`MINIMAX_N_GPU=1`), lower resolution (`MINIMAX_WIDTH=1024 MINIMAX_HEIGHT=576`), or reduce `DLO_RESIDENT_LAYERS`.

**GPU container spins up during download** — Set `MINIMAX_MIN_CONTAINERS=0` (default). The GPU class only starts when receiving requests.

**"FlashAttention kernel not found"** — H100 uses FA2/FA3 fallback (FA4 is Blackwell-only). This is expected and handled automatically.

## References

- [MiniMax H3 Model Card](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [vLLM-Omni Video API Docs](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/videos_api/)
- [vLLM Recipes: MiniMax H3](https://recipes.vllm.ai/MiniMaxAI/MiniMax-H3)
