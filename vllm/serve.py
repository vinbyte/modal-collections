"""
vLLM on Modal — Deploys an OpenAI-compatible LLM inference server using vLLM.

Architecture:
  - Modal builds a container image with vLLM and transformers.
  - At container startup, vLLM serves the configured model on an OpenAI-compatible
    API endpoint (/v1/chat/completions, /v1/completions, etc.).
  - Model weights and vLLM JIT cache are stored in persistent Modal Volumes
    so they don't need to be re-downloaded on every cold start.
  - The server is exposed via Modal's @modal.web_server decorator.

Usage:
  modal serve vllm/serve.py    # dev mode, logs in terminal
  modal deploy vllm/serve.py   # production, persistent

Client (after deploy):
  pip install openai
  export OPENAI_BASE_URL="<url-from-modal>/v1"
  export OPENAI_API_KEY="not-needed"
  python -c "from openai import OpenAI; c=OpenAI(); print(c.chat.completions.create(model='Qwen/Qwen3.6-27B', messages=[{'role':'user','content':'Hello!'}]))"
"""

import json
import logging
import subprocess
import sys
import time

import modal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("vllm")

# ── Runtime config ──────────────────────────────────────────────────────────
# Available GPU types: T4, L4, A10, L40S, A100, A100-40GB, A100-80GB,
#   RTX-PRO-6000, H100, H100!, H200, B200, B200+
# See https://modal.com/pricing for details.
GPU_TYPE = "A100-80GB"
N_GPU = 1
MIN_CONTAINERS = 1  # min number of container replicas to keep warm (0 = scale to zero)
TIMEOUT = 600  # max seconds a container stays alive (10 min)
SCALEDOWN_WINDOW = 900  # seconds to stay up with no requests (15 min)
STARTUP_TIMEOUT = 600  # seconds Modal waits for the container to be ready
MAX_CONCURRENT_INPUTS = 100  # max concurrent requests per container replica
VLLM_PORT = 8000  # port vLLM listens on inside the container

# ── Model config ────────────────────────────────────────────────────────────
MODEL_NAME = "cyankiwi/Qwen3.6-27B-AWQ-INT4"
MODEL_REVISION = None  # pin a specific commit hash to avoid surprises, e.g. "abc1234"

# ── vLLM engine config ──────────────────────────────────────────────────────
FAST_BOOT = False  # True = skip CUDA graph capture & torch compile (faster cold start, slower inference)
MAX_MODEL_LEN = 262144  # max context length in tokens (AWQ INT4 = ~21 GB model, fits 262K on A100-80GB)
LANGUAGE_MODEL_ONLY = False  # False = vision encoder active (multimodal); True = text-only (saves ~1.2 GB VRAM)
REASONING_PARSER = (
    "qwen3"  # reasoning parser for thinking mode ("qwen3", "deepseek_r1", etc.)
)
ENABLE_AUTO_TOOL_CHOICE = True  # enable tool use support
TOOL_CALL_PARSER = "qwen3_coder"  # tool call parser name (e.g. "qwen3_coder", "hermes")

# ── Persistent storage ──────────────────────────────────────────────────────
HF_CACHE_VOL_NAME = "huggingface-cache"  # Modal Volume for HF model weights
VLLM_CACHE_VOL_NAME = "vllm-cache"  # Modal Volume for vLLM JIT cache
HF_CACHE_MOUNT = "/root/.cache/huggingface"
VLLM_CACHE_MOUNT = "/root/.cache/vllm"

# ── Dependency versions ──────────────────────────────────────────────────────
VLLM_VERSION = "0.19.0"
TRANSFORMERS_VERSION = "5.5.0"  # needed for Qwen3.6 support in vLLM 0.19.0
CUDA_BASE_IMAGE = "nvidia/cuda:12.9.0-devel-ubuntu22.04"
PYTHON_VERSION = "3.12"

# ── Container image ──────────────────────────────────────────────────────────
vllm_image = (
    modal.Image.from_registry(CUDA_BASE_IMAGE, add_python=PYTHON_VERSION)
    .entrypoint([])
    .uv_pip_install(f"vllm=={VLLM_VERSION}")
    .uv_pip_install(f"transformers=={TRANSFORMERS_VERSION}")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

# ── Persistent volumes ──────────────────────────────────────────────────────
hf_cache_vol = modal.Volume.from_name(HF_CACHE_VOL_NAME, create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name(VLLM_CACHE_VOL_NAME, create_if_missing=True)

app = modal.App("vllm-inference")


@app.function(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=TIMEOUT,
    volumes={
        HF_CACHE_MOUNT: hf_cache_vol,
        VLLM_CACHE_MOUNT: vllm_cache_vol,
    },
    min_containers=MIN_CONTAINERS,
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_INPUTS)
@modal.web_server(port=VLLM_PORT, startup_timeout=STARTUP_TIMEOUT)
def serve():
    """Start the vLLM OpenAI-compatible server as a subprocess.

    The server exposes:
      - POST /v1/chat/completions  — chat completion (streaming supported)
      - POST /v1/completions       — text completion
      - GET  /v1/models            — list available models
      - GET  /health               — health check

    Once deployed, Modal provides a stable URL. You can use any OpenAI-compatible
    client (openai SDK, curl, etc.) to interact with it.
    """
    t_start = time.monotonic()
    logger.info("=== vLLM server startup ===")
    logger.info(
        "Model: %s | GPU: %s x %d | Max ctx: %d tokens",
        MODEL_NAME,
        GPU_TYPE,
        N_GPU,
        MAX_MODEL_LEN,
    )
    logger.info(
        "Fast boot: %s | Reasoning parser: %s | Port: %d",
        FAST_BOOT,
        REASONING_PARSER,
        VLLM_PORT,
    )
    _log_gpu_info()

    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--tensor-parallel-size",
        str(N_GPU),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--uvicorn-log-level",
        "info",
        "--async-scheduling",
    ]

    if MODEL_REVISION:
        cmd += ["--revision", MODEL_REVISION]

    if FAST_BOOT:
        cmd += ["--enforce-eager"]
    else:
        cmd += ["--no-enforce-eager"]

    if REASONING_PARSER:
        cmd += ["--reasoning-parser", REASONING_PARSER]

    if LANGUAGE_MODEL_ONLY:
        cmd += ["--language-model-only"]

    if ENABLE_AUTO_TOOL_CHOICE:
        cmd += ["--enable-auto-tool-choice"]
        if TOOL_CALL_PARSER:
            cmd += ["--tool-call-parser", TOOL_CALL_PARSER]

    logger.info("Command: %s", " ".join(cmd))

    proc = subprocess.Popen(
        " ".join(cmd),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    logger.info("vLLM PID: %d", proc.pid)
    _stream_output(proc)

    elapsed = time.monotonic() - t_start
    logger.info("=== vLLM serve() returned in %.1fs ===", elapsed)


# ── Local entrypoint (testing) ──────────────────────────────────────────────


@app.local_entrypoint()
async def main(
    content: str = "Explain the singular value decomposition in simple terms.",
):
    """Test the deployed server by sending a chat completion request.

    Usage:
      modal run vllm/vllm.py
      modal run vllm/vllm.py --content "What is the meaning of life?"
    """
    import aiohttp

    url = await serve.get_web_url.aio()
    logger.info("Server URL: %s", url)

    messages = [
        {"role": "system", "content": "You are a helpful and concise assistant."},
        {"role": "user", "content": content},
    ]

    async with aiohttp.ClientSession(base_url=url) as session:
        logger.info("Running health check...")
        async with session.get(
            "/health", timeout=aiohttp.ClientTimeout(total=300)
        ) as resp:
            assert resp.status == 200, f"Health check failed: {resp.status}"
            logger.info("Health check passed")

        logger.info("Sending chat request: %s", content)
        payload = {
            "messages": messages,
            "model": MODEL_NAME,
            "stream": True,
            "max_tokens": 4096,
        }
        if REASONING_PARSER:
            payload["chat_template_kwargs"] = {"enable_thinking": True}

        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

        async with session.post(
            "/v1/chat/completions", json=payload, headers=headers
        ) as resp:
            resp.raise_for_status()
            async for raw in resp.content:
                line = raw.decode().strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    line = line[len("data: ") :]
                chunk = json.loads(line)
                delta = chunk["choices"][0]["delta"]
                text = (
                    delta.get("content")
                    or delta.get("reasoning")
                    or delta.get("reasoning_content")
                )
                if text:
                    print(text, end="", flush=True)
            print()


# ── Helpers ──────────────────────────────────────────────────────────────────


def _stream_output(proc):
    """Stream vLLM subprocess output to structured logs in a daemon thread."""
    import threading

    def _reader():
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            if any(kw in line.lower() for kw in ("error", "exception", "traceback")):
                logger.error("VLLM: %s", line)
            elif any(
                kw in line.lower() for kw in ("loaded", "ready", "serving", "uvicorn")
            ):
                logger.info("VLLM: %s", line)
            elif any(kw in line.lower() for kw in ("download", "loading", "model")):
                logger.info("VLLM: %s", line)
            else:
                logger.debug("VLLM: %s", line)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()


def _log_gpu_info():
    """Log GPU hardware info for debugging."""
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
