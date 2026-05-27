"""
llama.cpp on Modal — Deploys an OpenAI-compatible LLM inference server using llama.cpp.

Architecture:
  - Modal builds a container image from the official llama.cpp CUDA Docker image.
  - At container startup, llama-server serves the configured model on an
    OpenAI-compatible API endpoint (/v1/chat/completions, /v1/completions, etc.).
  - Model weights are downloaded from Hugging Face and persisted in a Modal Volume
    so they don't need to be re-downloaded on every cold start.
  - The server is exposed via Modal's @modal.web_server decorator.

Usage:
  modal serve llamacpp/serve.py    # dev mode, logs in terminal
  modal deploy llamacpp/serve.py   # production, persistent

Client (after deploy):
  pip install openai
  export OPENAI_BASE_URL="<url-from-modal>/v1"
  export OPENAI_API_KEY="not-needed"
  python -c "from openai import OpenAI; c=OpenAI(); print(c.chat.completions.create(model='gemma-3-4b-it', messages=[{'role':'user','content':'Hello!'}]))"
"""

import json
import logging
import subprocess
import time

import modal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("llamacpp")

# ── Runtime config ──────────────────────────────────────────────────────────
# Available GPU types: T4, L4, A10, L40S, A100, A100-40GB, A100-80GB,
#   RTX-PRO-6000, H100, H100!, H200, B200, B200+
# See https://modal.com/pricing for details.
GPU_TYPE = "L4"
N_GPU = 1
MIN_CONTAINERS = 1
TIMEOUT = 600
SCALEDOWN_WINDOW = 900
STARTUP_TIMEOUT = 600
MAX_CONCURRENT_INPUTS = 4
LLAMACPP_PORT = 8080
ENABLE_MEMORY_SNAPSHOT = True
ENABLE_GPU_SNAPSHOT = True

# ── Model config ────────────────────────────────────────────────────────────
HF_REPO = "Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF"
HF_FILE = "Qwen3.5-9B.Q4_K_M.gguf"  # specific GGUF filename, e.g. "gemma-3-4b-it-Q4_K_M.gguf"
MODEL_ALIAS = "Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled"

# ── llama.cpp engine config ─────────────────────────────────────────────────
CTX_SIZE = 32768
GPU_LAYERS = 99  # layers on GPU (99 = all, "auto" = auto-fit)
THREADS = -1  # CPU threads (-1 = auto)
BATCH_SIZE = 2048
UBATCH_SIZE = 1024
FLASH_ATTN = True
CONT_BATCHING = True
N_PARALLEL = 4
CACHE_TYPE_K = "q8_0"  # KV cache quantization for K (saves VRAM)
CACHE_TYPE_V = "q8_0"
REASONING_FORMAT = "auto"  # "none", "deepseek", "auto"
CHAT_TEMPLATE = None  # custom jinja template (None = model default)
JSON_SCHEMA = None  # path to JSON schema file for constrained output

# ── Persistent storage ──────────────────────────────────────────────────────
HF_CACHE_VOL_NAME = "huggingface-cache"
HF_CACHE_MOUNT = "/root/.cache/huggingface"

# ── Container image ──────────────────────────────────────────────────────────
LLAMACPP_DOCKER_IMAGE = "ghcr.io/ggml-org/llama.cpp:server-cuda"
PYTHON_VERSION = "3.12"

llamacpp_image = (
    modal.Image.from_registry(LLAMACPP_DOCKER_IMAGE, add_python=PYTHON_VERSION)
    .entrypoint([])
    .run_commands("pip install aiohttp")
    .env({
        "HF_HOME": "/root/.cache/huggingface",
        "LLAMA_CACHE": "/root/.cache/huggingface/hub",
    })
)

# ── Persistent volumes ──────────────────────────────────────────────────────
hf_cache_vol = modal.Volume.from_name(HF_CACHE_VOL_NAME, create_if_missing=True)

app = modal.App("llamacpp-inference")


@app.cls(
    image=llamacpp_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=TIMEOUT,
    volumes={HF_CACHE_MOUNT: hf_cache_vol},
    min_containers=MIN_CONTAINERS,
    enable_memory_snapshot=ENABLE_MEMORY_SNAPSHOT,
    experimental_options={"enable_gpu_snapshot": ENABLE_GPU_SNAPSHOT},
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_INPUTS)
class LlamaCpp:
    """Modal class that runs llama.cpp as an OpenAI-compatible inference server.

    Lifecycle:
      1. @modal.enter(snap=True)  start_checkpoint() — log GPU info,
         launch llama-server subprocess, wait for model load, take GPU snapshot
      2. @modal.enter(snap=False) start_restore() — kill frozen server, restart it
      3. @modal.web_server(8080)  serve() — placeholder, server runs as subprocess
      4. @modal.exit()            cleanup() — terminate subprocess
    """

    @modal.enter(snap=True)
    def start_checkpoint(self):
        """Cold start: launch llama-server, wait for model load, take GPU snapshot."""
        t_start = time.monotonic()
        logger.info("=== llama.cpp GPU snapshot cold start ===")
        logger.info(
            "HF repo: %s | GPU: %s x %d | Context: %d tokens | GPU layers: %s",
            HF_REPO, GPU_TYPE, N_GPU, CTX_SIZE, GPU_LAYERS,
        )
        _log_gpu_info()
        self._launch_server()
        elapsed = time.monotonic() - t_start
        logger.info("=== cold start complete in %.1fs ===", elapsed)

    @modal.enter(snap=False)
    def start_restore(self):
        """Warm start: restart server after GPU snapshot restore.

        GPU snapshots can leave event-loop subprocesses frozen in place.
        Kill the frozen server and restart it to recover.
        """
        logger.info("=== llama.cpp restore from GPU snapshot ===")
        self._terminate_server()
        self._launch_server()
        logger.info("=== restore complete ===")

    @modal.web_server(port=LLAMACPP_PORT, startup_timeout=STARTUP_TIMEOUT)
    def serve(self):
        """Placeholder — llama-server runs as a subprocess."""
        logger.info("=== llama.cpp web_server active ===")

    @modal.exit()
    def cleanup(self):
        """Terminate llama-server subprocess."""
        self._terminate_server()

    # ── Subprocess management ────────────────────────────────────────────────

    def _build_cmd(self):
        """Build the llama-server command-line argument list."""
        cmd = [
            "/app/llama-server",
            "--hf-repo", HF_REPO,
            "--alias", MODEL_ALIAS,
            "--host", "0.0.0.0",
            "--port", str(LLAMACPP_PORT),
            "--ctx-size", str(CTX_SIZE),
            "--gpu-layers", str(GPU_LAYERS),
            "--threads", str(THREADS),
            "--batch-size", str(BATCH_SIZE),
            "--ubatch-size", str(UBATCH_SIZE),
            "--parallel", str(N_PARALLEL),
            "--cache-type-k", CACHE_TYPE_K,
            "--cache-type-v", CACHE_TYPE_V,
            "--reasoning-format", REASONING_FORMAT,
            "--log-verbose",
        ]

        if HF_FILE:
            cmd += ["--hf-file", HF_FILE]

        cmd += ["--flash-attn", "on"] if FLASH_ATTN else ["--flash-attn", "off"]

        if CONT_BATCHING:
            cmd += ["--cont-batching"]
        else:
            cmd += ["--no-cont-batching"]

        if CHAT_TEMPLATE:
            cmd += ["--chat-template", CHAT_TEMPLATE]

        if JSON_SCHEMA:
            cmd += ["--json-schema-file", JSON_SCHEMA]

        return cmd

    def _launch_server(self):
        """Start llama-server as a subprocess and wait until model is ready.

        Exposes OpenAI-compatible endpoints:
          - POST /v1/chat/completions  — chat completion (streaming supported)
          - POST /v1/completions       — text completion
          - GET  /v1/models            — list available models
          - GET  /health               — health check
        """
        cmd = self._build_cmd()
        logger.info("Command: %s", " ".join(cmd))

        self.server_proc = subprocess.Popen(
            " ".join(cmd),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        logger.info("llama-server PID: %d", self.server_proc.pid)

        _wait_until_ready(self.server_proc)

    def _terminate_server(self):
        """Kill the frozen llama-server subprocess from a previous snapshot."""
        proc = getattr(self, "server_proc", None)
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=10)
                logger.info("Terminated old llama-server (PID: %d)", proc.pid)
            except (ProcessLookupError, OSError, subprocess.TimeoutExpired):
                try:
                    proc.kill()
                except Exception:
                    pass


# ── Local entrypoint (testing) ──────────────────────────────────────────────


@app.local_entrypoint()
async def main(
    content: str = "Explain the singular value decomposition in simple terms.",
):
    """Test the deployed server by sending a chat completion request.

    Usage:
      modal run llamacpp/serve.py
      modal run llamacpp/serve.py --content "What is the meaning of life?"
    """
    import aiohttp

    url = await LlamaCpp.serve.get_web_url.aio()
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
            "model": MODEL_ALIAS,
            "stream": True,
            "max_tokens": 4096,
        }

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
                    line = line[len("data: "):]
                chunk = json.loads(line)
                delta = chunk["choices"][0]["delta"]
                text = (
                    delta.get("content")
                    or delta.get("reasoning_content")
                    or delta.get("reasoning")
                )
                if text:
                    print(text, end="", flush=True)
            print()


# ── Helpers ──────────────────────────────────────────────────────────────────


def _wait_until_ready(proc):
    """Read subprocess stdout until the model is loaded, then keep streaming in background."""
    import threading

    ready = threading.Event()

    def _reader():
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            logger.info("LLAMACPP: %s", line)
            if any(kw in line.lower() for kw in ("server is listening", "http server listening")):
                ready.set()
        ready.set()  # process exited or stdout closed — unblock caller

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    logger.info("Waiting for llama-server to finish loading model...")
    if not ready.wait(timeout=STARTUP_TIMEOUT):
        logger.error("llama-server did not become ready within %ds", STARTUP_TIMEOUT)
    else:
        logger.info("llama-server is ready and listening")


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
