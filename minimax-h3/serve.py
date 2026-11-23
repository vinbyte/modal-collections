"""
MiniMax H3 on Modal — Deploys multimodal video+audio generation via vLLM-Omni.

Architecture:
  - Uses vllm/vllm-omni:minimax-h3 Docker image with pre-built FA4 kernels
  - Model weights downloaded from HuggingFace (gated) to a persistent volume
  - vllm serve --omni exposes OpenAI-compatible video API on port 8000
  - Supports two partitions: FL2VA (t2va, fl2va) and Ref2VA (ref2va)
  - Configurable GPU count: 1 (offload), 2 (DLO), 4 (best), 8 (max)
  - GPU snapshot lifecycle for fast cold boots (~10x faster)

API endpoints:
  POST /v1/videos/sync  — synchronous video generation (returns MP4)
  POST /v1/videos       — async video generation (job polling)
  GET  /health          — health check

Prerequisites:
  1. Get HuggingFace access: https://huggingface.co/MiniMaxAI/MiniMax-H3
   2. Create Modal secret: modal secret create huggingface-secret HF_TOKEN=<your-hf-token>

Usage:
  # Step 1: Download model (CPU-only, no GPU cost)
  modal run minimax-h3/serve.py::download_model

  # Step 2: Serve (GPU)
  modal serve minimax-h3/serve.py    # dev mode, logs in terminal
  modal deploy minimax-h3/serve.py   # production, persistent

Client (after deploy):
  export BASE_URL="<url-from-modal>"
  curl -X POST "$BASE_URL/v1/videos/sync" \
    -F "prompt=At night, three cats march into a bedroom playing tiny brass instruments." \
    -F "width=1344" -F "height=768" -F "fps=24" -F "num_inference_steps=50" \
    -F "flow_shift=12" -F "seed=1101" \
    -F 'extra_params={"task":"t2va","duration":5.0,"audio_flow_shift":3.0}' \
    -o output.mp4
"""

import json
import logging
import os
import subprocess
import time
from pathlib import Path

import modal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("minimax-h3")

# ═══════════════════════════════════════════════════════════════════════════════
# Runtime config
# ═══════════════════════════════════════════════════════════════════════════════
# GPU types: H100, H100!, A100-80GB, H200, B200, B200+
# H100 = PCIe version (80 GB), H100! = SXM version (higher bandwidth)
# See https://modal.com/pricing for details and availability.
GPU_TYPE = os.environ.get("MINIMAX_GPU_TYPE", "B200")
N_GPU = int(os.environ.get("MINIMAX_N_GPU", "4"))  # 1, 2, 4, or 8

# Host RAM override — only needed for offload paths (1/2 GPU).
# 4+ GPU TP-only mode fits in VRAM and uses minimal host RAM.
_HOST_RAM_MIB = int(os.environ.get("MINIMAX_HOST_RAM_MIB", "0"))
if _HOST_RAM_MIB > 0:
    HOST_MEMORY = _HOST_RAM_MIB
elif N_GPU <= 2:
    HOST_MEMORY = 393216  # 384 GiB for CPU offload / DLO
else:
    HOST_MEMORY = None  # let Modal pick default

MIN_CONTAINERS = int(os.environ.get("MINIMAX_MIN_CONTAINERS", "0"))
TIMEOUT = 3600  # video generation takes minutes
SCALEDOWN_WINDOW = 900
STARTUP_TIMEOUT = 1200  # model loading + CUDA graph capture can take minutes
MAX_CONCURRENT_INPUTS = 1  # H3 processes one generation per batch
VLLM_PORT = 8000

ENABLE_MEMORY_SNAPSHOT = False
ENABLE_GPU_SNAPSHOT = False

# ═══════════════════════════════════════════════════════════════════════════════
# Model config
# ═══════════════════════════════════════════════════════════════════════════════
PARTITION = os.environ.get("MINIMAX_PARTITION", "FL2VA")  # FL2VA or Ref2VA
MODEL_REPO = "MiniMaxAI/MiniMax-H3"

# ═══════════════════════════════════════════════════════════════════════════════
# Generation defaults
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULT_WIDTH = int(os.environ.get("MINIMAX_WIDTH", "1440"))
DEFAULT_HEIGHT = int(os.environ.get("MINIMAX_HEIGHT", "2560"))
DEFAULT_FPS = 24
DEFAULT_STEPS = 50
DEFAULT_FLOW_SHIFT = 12
DEFAULT_AUDIO_FLOW_SHIFT = 3.0
DEFAULT_DURATION = 5.0

# ═══════════════════════════════════════════════════════════════════════════════
# 2-GPU DLO (distributed layerwise offload) config
# ═══════════════════════════════════════════════════════════════════════════════
DLO_RESIDENT_LAYERS = int(os.environ.get("MINIMAX_DLO_RESIDENT_LAYERS", "20"))

# ═══════════════════════════════════════════════════════════════════════════════
# Persistent storage
# ═══════════════════════════════════════════════════════════════════════════════
VOL_NAME = "minimax-h3-models"
VOL_MOUNT = "/mnt/models"
HF_CACHE_DIR = os.path.join(VOL_MOUNT, ".hf-cache")
MODEL_LOCAL_DIR = os.path.join(VOL_MOUNT, "MiniMax-H3")

# ═══════════════════════════════════════════════════════════════════════════════
# Container image
# ═══════════════════════════════════════════════════════════════════════════════
# vllm/vllm-omni:minimax-h3 bundles: vLLM-Omni + MiniMax H3 handlers + FA4 kernels
vllm_omni_image = (
    modal.Image.from_registry("vllm/vllm-omni:minimax-h3")
    .entrypoint([])
    .env(
        {
            "HF_HOME": HF_CACHE_DIR,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
)

# ═══════════════════════════════════════════════════════════════════════════════
# Persistent volumes & secrets
# ═══════════════════════════════════════════════════════════════════════════════
hf_cache_vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

HF_SECRET = modal.Secret.from_name(
    "huggingface-secret",
    required_keys=["HF_TOKEN"],
)

app = modal.App("minimax-h3")


# ═══════════════════════════════════════════════════════════════════════════════
# CPU-only model download function (no GPU cost!)
# ═══════════════════════════════════════════════════════════════════════════════


@app.function(
    image=vllm_omni_image,
    volumes={VOL_MOUNT: hf_cache_vol},
    secrets=[HF_SECRET],
    timeout=7200,
)
def download_model():
    """Download MiniMax H3 from HuggingFace to persistent volume (CPU-only, no GPU cost).

    Usage: modal run minimax-h3/serve.py::download_model
    """
    partition_path = os.path.join(MODEL_LOCAL_DIR, PARTITION)
    done_marker = os.path.join(MODEL_LOCAL_DIR, ".download_complete")

    if os.path.isfile(done_marker) and os.path.isdir(partition_path):
        logger.info("Model already downloaded: %s", partition_path)
        return

    os.makedirs(MODEL_LOCAL_DIR, exist_ok=True)

    logger.info(
        "Downloading %s to %s (gated — HF_TOKEN required)",
        MODEL_REPO, MODEL_LOCAL_DIR,
    )
    cmd = [
        "hf", "download", MODEL_REPO,
        "--local-dir", MODEL_LOCAL_DIR,
    ]
    logger.info("Command: %s", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "HF_HOME": HF_CACHE_DIR},
    )
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        if any(kw in line.lower() for kw in ("error", "fail", "traceback", "denied")):
            logger.error("HF: %s", line)
        elif any(kw in line.lower() for kw in ("downloading", "fetching", "complete", "%", "mb", "gb", "file", "resolving")):
            logger.info("HF: %s", line)
        else:
            logger.info("HF: %s", line)

    proc.wait(timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Model download failed with exit code {proc.returncode}. Check HF_TOKEN."
        )

    logger.info("Model downloaded successfully")

    if os.path.isdir(partition_path):
        Path(done_marker).touch()
        logger.info(
            "Partition %s ready (%d files)",
            PARTITION,
            len(list(Path(partition_path).rglob("*"))),
        )
    else:
        available = os.listdir(MODEL_LOCAL_DIR)
        raise RuntimeError(
            f"Partition {PARTITION} not found after download. Available: {available}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# GPU-powered inference server
# ═══════════════════════════════════════════════════════════════════════════════


@app.cls(
    image=vllm_omni_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    memory=HOST_MEMORY,
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=TIMEOUT,
    volumes={
        VOL_MOUNT: hf_cache_vol,
    },
    secrets=[HF_SECRET],
    min_containers=MIN_CONTAINERS,
    enable_memory_snapshot=ENABLE_MEMORY_SNAPSHOT,
    experimental_options={"enable_gpu_snapshot": ENABLE_GPU_SNAPSHOT},
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_INPUTS)
class MiniMaxH3:
    """Modal class that serves MiniMax H3 via vLLM-Omni.

    Lifecycle:
      1. Run download_model() first to cache model on volume (CPU-only)
      2. @modal.enter(snap=True)  start_checkpoint() — verify model exists,
         launch vLLM serve subprocess, wait for ready, take GPU snapshot
      3. @modal.enter(snap=False) start_restore() — kill frozen vLLM,
         restart it (uvicorn event loops stall after GPU snapshot restore)
      4. @modal.web_server(8000)  serve() — placeholder, vLLM runs as subprocess
      5. @modal.exit()            cleanup() — terminate vLLM subprocess
    """

    @modal.enter()
    def start(self):
        """Cold start: launch vLLM server."""
        import torch

        t_start = time.monotonic()
        logger.info("=== MiniMax H3 cold start ===")
        logger.info(
            "Model: %s | Partition: %s | GPU: %s x %d | Resolution: %dx%d",
            MODEL_REPO, PARTITION, GPU_TYPE, torch.cuda.device_count() or N_GPU, DEFAULT_WIDTH, DEFAULT_HEIGHT,
        )
        _log_gpu_info()

        model_path = os.path.join(MODEL_LOCAL_DIR, PARTITION)
        done_marker = os.path.join(MODEL_LOCAL_DIR, ".download_complete")
        if not (os.path.isfile(done_marker) and os.path.isdir(model_path)):
            raise RuntimeError(
                f"Model not ready at {model_path}. "
                f"Run download first: modal run minimax-h3/serve.py::download_model"
            )

        self._launch_vllm()

        elapsed = time.monotonic() - t_start
        logger.info("=== cold start complete in %.1fs ===", elapsed)

    @modal.web_server(port=VLLM_PORT, startup_timeout=STARTUP_TIMEOUT)
    def serve(self):
        """Placeholder — vLLM-Omni runs as a subprocess."""
        logger.info("=== vLLM-Omni web_server active ===")

    @modal.exit()
    def cleanup(self):
        """Terminate vLLM subprocess."""
        self._terminate_vllm()

    # ── vLLM subprocess management ─────────────────────────────────────────

    def _launch_vllm(self):
        """Start vLLM-Omni server as a subprocess.

        Exposes:
          - POST /v1/videos/sync — synchronous video generation
          - POST /v1/videos      — async video generation (job polling)
          - GET  /health         — health check
        """
        model_path = os.path.join(MODEL_LOCAL_DIR, PARTITION)
        if not os.path.isdir(model_path):
            raise RuntimeError(
                f"Model partition not found: {model_path}. "
                f"Run download first: modal run minimax-h3/serve.py::download_model"
            )

        cmd = self._build_vllm_cmd(model_path)
        cmd_str = " ".join(cmd)

        env = os.environ.copy()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        env["VLLM_OMNI_VIDEO_SYNC_TIMEOUT"] = str(TIMEOUT)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        logger.info("=== Launching vLLM-Omni ===")
        logger.info("Model path: %s", model_path)
        logger.info("Command: %s", cmd_str)

        self.vllm_proc = subprocess.Popen(
            cmd_str,
            shell=True,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        logger.info("vLLM PID: %d", self.vllm_proc.pid)
        _stream_output(self.vllm_proc)

    def _build_vllm_cmd(self, model_path: str):
        """Build the vLLM serve CLI command for the configured GPU count.

        Uses torch.cuda.device_count() to auto-detect how many GPUs Modal allocated.
        Falls back to N_GPU module-level constant.

        Strategy per GPU count:
          1 GPU  — model-level CPU offload, accuracy-qualified reference path
          2 GPU  — distributed layerwise offload (DLO), needs ~384 GB host RAM
          4 GPU  — tensor parallelism (TP4), shards DiT + encoder across GPUs
          8 GPU  — tensor parallelism (TP8), max throughput
        """
        import torch

        ng = torch.cuda.device_count() or N_GPU
        logger.info("Detected %d GPU(s)", ng)
        cmd = [
            "vllm",
            "serve",
            model_path,
            "--omni",
            "--host", "0.0.0.0",
            "--port", str(VLLM_PORT),
            "--trust-remote-code",
            "--num-gpus", str(ng),
            "--init-timeout", "3600",
        ]

        if ng == 1:
            cmd += [
                "--enable-cpu-offload",
                "--diffusion-attention-backend", "FLASH_ATTN",
            ]

        elif ng == 2:
            cmd += [
                "--tensor-parallel-size", "2",
                "--usp", "1",
                "--ring", "1",
                "--text-encoder-tp-size", "2",
                "--vae-patch-parallel-size", "2",
                "--vae-parallel-mode", "tile",
                "--vae-use-tiling",
                "--enable-distributed-layerwise-offload",
                "--dlo-no-use-allgather",
                "--dlo-resident-layers", str(DLO_RESIDENT_LAYERS),
                "--enforce-eager",
                "--diffusion-attention-backend", "CUDNN_ATTN",
            ]

        elif ng == 4:
            cmd += [
                "--usp", "4",
                "--ring", "1",
                "--text-encoder-tp-size", "4",
                "--vae-patch-parallel-size", "4",
                "--vae-parallel-mode", "tile",
                "--vae-use-tiling",
                "--diffusion-attention-backend", "FLASH_ATTN",
            ]

        elif ng == 8:
            cmd += [
                "--usp", "8",
                "--ring", "1",
                "--text-encoder-tp-size", "8",
                "--vae-patch-parallel-size", "8",
                "--vae-parallel-mode", "tile",
                "--vae-use-tiling",
                "--diffusion-attention-backend", "FLASH_ATTN",
            ]

        else:
            raise ValueError(
                f"Unsupported N_GPU={ng}. Use 1, 2, 4, or 8."
            )

        return cmd

    def _terminate_vllm(self):
        """Kill the frozen vLLM subprocess from a previous snapshot."""
        proc = getattr(self, "vllm_proc", None)
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=10)
                logger.info("Terminated old vLLM process (PID: %d)", proc.pid)
            except (ProcessLookupError, OSError, subprocess.TimeoutExpired):
                try:
                    proc.kill()
                except Exception:
                    pass


# ═══════════════════════════════════════════════════════════════════════════════
# Local entrypoint (testing)
# ═══════════════════════════════════════════════════════════════════════════════


@app.local_entrypoint()
async def main(
    prompt: str = "At night, three cats march into a bedroom playing tiny brass instruments, then abruptly file out, with synchronized room ambience.",
    duration: float = DEFAULT_DURATION,
):
    """Test the deployed server with a text-to-video+audio generation request.

    Usage:
      modal run minimax-h3/serve.py
      modal run minimax-h3/serve.py --prompt "A golden retriever surfing on a rainbow wave"
      modal run minimax-h3/serve.py --duration 8.0
    """
    import aiohttp
    import asyncio

    url = await MiniMaxH3().serve.get_web_url.aio()
    logger.info("Server URL: %s", url)

    async with aiohttp.ClientSession(base_url=url) as session:
        logger.info("Waiting for server to be ready (up to %ds)...", STARTUP_TIMEOUT)
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            try:
                async with session.get(
                    "/health", timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        logger.info("Server is ready")
                        break
            except Exception:
                pass
            await asyncio.sleep(5)
        else:
            raise RuntimeError(f"Server did not become ready within {STARTUP_TIMEOUT}s")

        logger.info(
            "Sending t2va request: prompt=%r, duration=%ss, resolution=%dx%d",
            prompt, duration, DEFAULT_WIDTH, DEFAULT_HEIGHT,
        )
        form = aiohttp.FormData()
        form.add_field("prompt", prompt)
        form.add_field("width", str(DEFAULT_WIDTH))
        form.add_field("height", str(DEFAULT_HEIGHT))
        form.add_field("fps", str(DEFAULT_FPS))
        form.add_field("num_inference_steps", str(DEFAULT_STEPS))
        form.add_field("flow_shift", str(DEFAULT_FLOW_SHIFT))
        form.add_field("seed", "1101")
        form.add_field(
            "extra_params",
            json.dumps({
                "task": "t2va",
                "duration": duration,
                "audio_flow_shift": DEFAULT_AUDIO_FLOW_SHIFT,
            }),
        )

        t_req = time.monotonic()
        async with session.post(
            "/v1/videos/sync",
            data=form,
            timeout=aiohttp.ClientTimeout(total=1800),
        ) as resp:
            if resp.status == 200:
                output_path = "/tmp/minimax-h3-t2va.mp4"
                content = await resp.read()
                with open(output_path, "wb") as f:
                    f.write(content)
                elapsed = time.monotonic() - t_req
                logger.info(
                    "Video saved to %s (%.1f MB, %.1fs elapsed)",
                    output_path, len(content) / 1e6, elapsed,
                )
            else:
                body = await resp.text()
                logger.error(
                    "Video generation failed: HTTP %s — %s",
                    resp.status, body[:1000],
                )


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _stream_output(proc):
    """Stream vLLM-Omni subprocess output to structured logs in a daemon thread."""
    import threading

    def _reader():
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            if any(
                kw in line.lower()
                for kw in ("error", "exception", "traceback", "failed")
            ):
                logger.error("VLLM: %s", line)
            elif any(
                kw in line.lower()
                for kw in ("loaded", "ready", "serving", "uvicorn", "startup", "complete")
            ):
                logger.info("VLLM: %s", line)
            elif any(
                kw in line.lower()
                for kw in ("download", "loading", "model", "cuda", "memory", "denoise")
            ):
                logger.info("VLLM: %s", line)
            else:
                logger.debug("VLLM: %s", line)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()


def _log_gpu_info():
    """Log GPU hardware info from nvidia-smi for debugging."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,cuda_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                logger.info("GPU: %s", line)
    except FileNotFoundError:
        logger.error("nvidia-smi not found - no GPU detected!")
    except Exception as e:
        logger.warning("Could not query GPU info: %s", e)
