"""
SGLang on Modal - Deploys an OpenAI-compatible LLM inference server using SGLang.

Architecture:
  - Modal builds a CUDA container image with SGLang installed.
  - At container startup, SGLang serves the configured Hugging Face model on an
    OpenAI-compatible API endpoint.
  - Model weights are downloaded from Hugging Face and persisted in a Modal
    Volume so they don't need to be re-downloaded on every cold start.
  - The server is exposed via Modal's @modal.web_server decorator.

Usage:
  modal serve sglang/serve.py    # dev mode, logs in terminal
  modal deploy sglang/serve.py   # production, persistent

Client (after deploy):
  pip install openai
  export OPENAI_BASE_URL="<url-from-modal>/v1"
  export OPENAI_API_KEY="not-needed"
  python -c "from openai import OpenAI; c=OpenAI(); print(c.chat.completions.create(model='FastContext-1.0-4B-SFT', messages=[{'role':'user','content':'Hello!'}]))"
"""

import json
import logging
import subprocess
import time
import urllib.request

import modal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("sglang")

# -- Runtime config -----------------------------------------------------------
# Available GPU types: T4, L4, A10, L40S, A100, A100-40GB, A100-80GB,
#   RTX-PRO-6000, H100, H100!, H200, B200, B200+
# See https://modal.com/pricing for details.
GPU_TYPE = "L40S"
N_GPU = 1
MIN_CONTAINERS = 1
TIMEOUT = 600
SCALEDOWN_WINDOW = 900
STARTUP_TIMEOUT = 600
MAX_CONCURRENT_INPUTS = 4
SGLANG_PORT = 30000
ENABLE_MEMORY_SNAPSHOT = False
ENABLE_GPU_SNAPSHOT = True

# -- Model config -------------------------------------------------------------
MODEL_NAME = "microsoft/FastContext-1.0-4B-SFT"
MODEL_REVISION = None
MODEL_ALIAS = "FastContext-1.0-4B-SFT"

# -- SGLang engine config -----------------------------------------------------
CONTEXT_LENGTH = 262144
TP_SIZE = N_GPU
MEM_FRACTION_STATIC = 0.8
DTYPE = "bfloat16"
TRUST_REMOTE_CODE = True
TOOL_CALL_PARSER = "qwen"

# -- Persistent storage -------------------------------------------------------
HF_CACHE_VOL_NAME = "huggingface-cache"
HF_CACHE_MOUNT = "/root/.cache/huggingface"

# -- Dependency versions ------------------------------------------------------
CUDA_BASE_IMAGE = "nvidia/cuda:12.9.0-devel-ubuntu22.04"
PYTHON_VERSION = "3.12"

# -- Container image ----------------------------------------------------------
sglang_image = (
    modal.Image.from_registry(CUDA_BASE_IMAGE, add_python=PYTHON_VERSION)
    .entrypoint([])
    .apt_install("libnuma1")
    .uv_pip_install("sglang[all]")
    .uv_pip_install("aiohttp")
    .env({
        "HF_HOME": HF_CACHE_MOUNT,
        "HF_XET_HIGH_PERFORMANCE": "1",
    })
)

# -- Persistent volumes -------------------------------------------------------
hf_cache_vol = modal.Volume.from_name(HF_CACHE_VOL_NAME, create_if_missing=True)

app = modal.App("sglang-inference")


@app.cls(
    image=sglang_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=TIMEOUT,
    volumes={HF_CACHE_MOUNT: hf_cache_vol},
    min_containers=MIN_CONTAINERS,
    enable_memory_snapshot=ENABLE_MEMORY_SNAPSHOT,
    experimental_options={"enable_gpu_snapshot": ENABLE_GPU_SNAPSHOT},
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_INPUTS)
class SGLang:
    """Modal class that runs SGLang as an OpenAI-compatible inference server.

    Lifecycle:
      1. @modal.enter()          start() - log GPU info, launch SGLang
      2. @modal.web_server(30000) serve() - placeholder, server runs as subprocess
      3. @modal.exit()           cleanup() - terminate subprocess
    """

    @modal.enter()
    def start(self):
        """Start SGLang server and wait for the model to be ready."""
        t_start = time.monotonic()
        logger.info("=== SGLang cold start ===")
        logger.info(
            "Model: %s | GPU: %s x %d | Context: %d tokens",
            MODEL_NAME,
            GPU_TYPE,
            N_GPU,
            CONTEXT_LENGTH,
        )
        _log_gpu_info()
        self._launch_server()
        elapsed = time.monotonic() - t_start
        logger.info("=== cold start complete in %.1fs ===", elapsed)

    @modal.web_server(port=SGLANG_PORT, startup_timeout=STARTUP_TIMEOUT)
    def serve(self):
        """Placeholder - SGLang runs as a subprocess."""
        logger.info("=== SGLang web_server active ===")

    @modal.exit()
    def cleanup(self):
        """Terminate SGLang subprocess."""
        self._terminate_server()

    # -- Subprocess management ------------------------------------------------

    def _build_cmd(self):
        """Build the SGLang OpenAI server command-line argument list."""
        cmd = [
            "python3", "-m", "sglang.launch_server",
            "--model-path", MODEL_NAME,
            "--served-model-name", MODEL_ALIAS,
            "--host", "0.0.0.0",
            "--port", str(SGLANG_PORT),
            "--tp-size", str(TP_SIZE),
            "--context-length", str(CONTEXT_LENGTH),
            "--mem-fraction-static", str(MEM_FRACTION_STATIC),
            "--dtype", DTYPE,
        ]

        if TRUST_REMOTE_CODE:
            cmd += ["--trust-remote-code"]

        if TOOL_CALL_PARSER:
            cmd += ["--tool-call-parser", TOOL_CALL_PARSER]

        if MODEL_REVISION:
            cmd += ["--revision", MODEL_REVISION]

        return cmd

    def _launch_server(self):
        """Start SGLang as a subprocess and wait until the model is ready.

        Exposes OpenAI-compatible endpoints:
          - POST /v1/chat/completions  - chat completion (streaming supported)
          - POST /v1/completions       - text completion
          - GET  /v1/models            - list available models
          - GET  /health               - health check
        """
        cmd = self._build_cmd()
        logger.info("Command: %s", " ".join(cmd))

        self.server_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        logger.info("SGLang PID: %d", self.server_proc.pid)

        _wait_until_ready(self.server_proc)

    def _terminate_server(self):
        """Kill the frozen SGLang subprocess from a previous snapshot."""
        proc = getattr(self, "server_proc", None)
        if proc is None or proc.poll() is not None:
            return

        logger.info("Terminating SGLang server (PID: %d)...", proc.pid)
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("SIGTERM timed out, sending SIGKILL to PID: %d", proc.pid)
            proc.kill()
            proc.wait(timeout=5)
        except (ProcessLookupError, OSError):
            pass
        logger.info("SGLang server terminated (PID: %d)", proc.pid)


# -- Local entrypoint (testing) ----------------------------------------------


@app.local_entrypoint()
async def main(
    content: str = "Explain the singular value decomposition in simple terms.",
):
    """Test the deployed server by sending a chat completion request.

    Usage:
      modal run sglang/serve.py
      modal run sglang/serve.py --content "What is the meaning of life?"
    """
    import aiohttp

    url = await SGLang.serve.get_web_url.aio()
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
                    line = line[len("data: ") :]
                chunk = json.loads(line)
                delta = chunk["choices"][0]["delta"]
                text = delta.get("content") or delta.get("reasoning_content")
                if text:
                    print(text, end="", flush=True)
            print()


# -- Helpers -----------------------------------------------------------------


def _wait_until_ready(proc):
    """Poll /health endpoint until the model is loaded and server is ready."""
    import threading

    dead = threading.Event()

    def _stream():
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                logger.info("SGLANG: %s", line)
        dead.set()

    t = threading.Thread(target=_stream, daemon=True)
    t.start()

    logger.info("Waiting for SGLang to finish loading model...")
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if dead.is_set():
            logger.error("SGLang exited prematurely (exit code: %s)", proc.poll())
            raise RuntimeError("SGLang exited before becoming ready")
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{SGLANG_PORT}/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    logger.info("SGLang is ready (health check passed)")
                    return
        except Exception:
            pass
        time.sleep(1)

    raise RuntimeError("SGLang did not become ready before timeout")


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
