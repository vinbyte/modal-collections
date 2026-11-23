"""Fish Speech's official API and Awesome WebUI on Modal.

modal run fish-speech/serve.py::download_model
modal serve fish-speech/serve.py
modal deploy fish-speech/serve.py
"""

import json
import logging
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

import modal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fish-speech")

# Keep backend, frontend, Python dependencies, and checkpoints reproducible.
SOURCE_REVISION = "0f8a12e8a651062fdee9c33a894bdccd3a8513f1"
MODELS = {
    "s2-pro": "1de9996b6be38b745688de084d87a5633f714e4e",
    "drama-3": "78e5ab79e49aa9a25abe8799eff2c4d73a1c3216",
}
MODEL = os.environ.get("FISH_MODEL", "s2-pro")
if MODEL not in MODELS:
    raise ValueError("FISH_MODEL must be s2-pro or drama-3")
MODEL_REVISION = MODELS[MODEL]
MODEL_REPO = f"fishaudio/{MODEL}"
GPU = os.environ.get("FISH_GPU", "H100" if MODEL == "drama-3" else "L40S")
COMPILE = os.environ.get("FISH_COMPILE", "0") == "1"
if MODEL == "drama-3":
    if COMPILE:
        raise ValueError("Drama 3 does not support FISH_COMPILE=1")
    if GPU not in {"L40S", "H100", "H100!", "H200", "B200"}:
        raise ValueError("Use an FP8-capable GPU with at least 48 GB for Drama 3")

MIN_CONTAINERS = int(os.environ.get("FISH_MIN_CONTAINERS", "0"))
SCALEDOWN_WINDOW = int(os.environ.get("FISH_SCALEDOWN_WINDOW", "300"))
HF_SECRET = os.environ.get("FISH_HF_SECRET", "")
ROOT = Path("/opt/fish-speech")
CHECKPOINT = Path("/models") / MODEL / MODEL_REVISION
PORT = 8080
STARTUP_TIMEOUT = 1200

app = modal.App("fish-speech")
models_volume = modal.Volume.from_name("fish-speech-models", create_if_missing=True)
references_volume = modal.Volume.from_name("fish-speech-references", create_if_missing=True)
cache_volume = modal.Volume.from_name("fish-speech-cache", create_if_missing=True)

# Capture deploy-time choices explicitly when Modal imports this module remotely.
runtime_env = {
    "FISH_MODEL": MODEL,
    "FISH_GPU": GPU,
    "FISH_COMPILE": "1" if COMPILE else "0",
    "FISH_MIN_CONTAINERS": str(MIN_CONTAINERS),
    "FISH_SCALEDOWN_WINDOW": str(SCALEDOWN_WINDOW),
    "FISH_HF_SECRET": HF_SECRET,
}
download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub==2.0.0")
    .env({**runtime_env, "HF_HOME": "/models/.hf-cache"})
)

# PyTorch's cu129 wheels include the CUDA runtime; Modal supplies the GPU driver.
# Node 22 is required by upstream's Vite 7 frontend. No Node/npm install locally.
image = (
    modal.Image.from_registry("node:22-bookworm-slim", add_python="3.12")
    .entrypoint([])
    .apt_install(
        # Modal's CPython selects clang when building PyAudio's native extension.
        "git", "build-essential", "clang", "cmake", "ffmpeg", "libsox-dev",
        "libasound2-dev", "portaudio19-dev", "libsndfile1",
    )
    .pip_install("uv==0.8.15")
    .run_commands(
        f"git init {ROOT}",
        f"git -C {ROOT} remote add origin https://github.com/fishaudio/fish-speech.git",
        f"git -C {ROOT} fetch --depth 1 origin {SOURCE_REVISION}",
        f"git -C {ROOT} checkout --detach FETCH_HEAD",
    )
    .workdir(ROOT)
    .run_commands("uv sync --python 3.12 --frozen --extra cu129")
    .add_local_file(
        Path(__file__).with_name("webui-layout.patch"),
        "/tmp/fish-speech-webui-layout.patch",
        copy=True,
    )
    .run_commands(
        "git apply --check /tmp/fish-speech-webui-layout.patch",
        "git apply /tmp/fish-speech-webui-layout.patch",
    )
    .run_commands(
        "npm --prefix awesome_webui ci --no-audit --no-fund",
        "npm --prefix awesome_webui run build",
        "test -s awesome_webui/dist/index.html",
    )
    .env({
        **runtime_env,
        "UV_NO_SYNC": "1",
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": "/cache/huggingface",
        "TORCH_HOME": "/cache/torch",
        "TORCHINDUCTOR_CACHE_DIR": "/cache/torchinductor",
        "TRITON_CACHE_DIR": "/cache/triton",
        "HF_HUB_OFFLINE": "1",
    })
)


def validate_checkpoint(checkpoint: Path) -> None:
    """Reject incomplete downloads before loading the model."""
    required = ["config.json", "tokenizer.json", "codec.pth", "model.safetensors.index.json"]
    index = checkpoint / "model.safetensors.index.json"
    if index.is_file():
        required.extend(set(json.loads(index.read_text())["weight_map"].values()))
    missing = [
        name for name in required
        if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(f"Incomplete checkpoint at {checkpoint}: {', '.join(sorted(missing))}")


@app.function(
    image=download_image,
    volumes={"/models": models_volume},
    secrets=[modal.Secret.from_name(HF_SECRET)] if HF_SECRET else [],
    timeout=7200,
    memory=2048,
    max_containers=1,
)
def download_model():
    """Download on CPU; interrupted downloads resume on the next invocation."""
    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    marker = CHECKPOINT / ".download_complete"
    marker.unlink(missing_ok=True)
    try:
        subprocess.run(
            ["hf", "download", MODEL_REPO, "--revision", MODEL_REVISION,
             "--local-dir", str(CHECKPOINT)],
            check=True,
        )
        validate_checkpoint(CHECKPOINT)
        marker.write_text(MODEL_REVISION)
    finally:
        # Retain partial files as well, so a retry does not start from scratch.
        models_volume.commit()
    logger.info("Checkpoint ready: %s", CHECKPOINT)


def server_command() -> list[str]:
    command = [
        str(ROOT / ".venv/bin/python"), "tools/api_server.py",
        "--listen", f"0.0.0.0:{PORT}",
        "--device", "cuda:0",
        "--llama-checkpoint-path", str(CHECKPOINT),
        "--decoder-checkpoint-path", str(CHECKPOINT / "codec.pth"),
        "--decoder-config-name", "modded_dac_vq",
        "--workers", "1",
        "--max-text-length", "5000",
    ]
    if COMPILE:
        command.append("--compile")
    return command


def wait_for_server(process: subprocess.Popen, timeout: float) -> None:
    """Wait for model initialization, and surface startup failures immediately."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Fish Speech exited with code {process.returncode}; see container logs")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/health", timeout=2) as response:
                if response.status == 200 and json.load(response).get("status") == "ok":
                    return
        except (OSError, urllib.error.URLError, ValueError):
            pass
        time.sleep(1)
    raise TimeoutError(f"Fish Speech was not ready after {timeout}s; see container logs")


def terminate_server(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


@app.cls(
    image=image,
    gpu=GPU,
    cpu=4,
    memory=98304 if MODEL == "drama-3" else 32768,
    volumes={
        "/models": models_volume,
        str(ROOT / "references"): references_volume,
        "/cache": cache_volume,
    },
    min_containers=MIN_CONTAINERS,
    max_containers=1,
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=1800,
    startup_timeout=STARTUP_TIMEOUT,
)
class FishSpeech:
    @modal.enter()
    def start(self):
        marker = CHECKPOINT / ".download_complete"
        if not marker.is_file() or marker.read_text() != MODEL_REVISION:
            raise RuntimeError(
                f"Download {MODEL} first: FISH_MODEL={MODEL} "
                "modal run fish-speech/serve.py::download_model"
            )
        validate_checkpoint(CHECKPOINT)
        logger.info("Starting %s on %s; Web UI: /ui", MODEL_REPO, GPU)
        self.process = subprocess.Popen(server_command(), cwd=ROOT)
        try:
            wait_for_server(self.process, STARTUP_TIMEOUT - 30)
        except Exception:
            terminate_server(self.process)
            raise

    @modal.web_server(PORT, startup_timeout=STARTUP_TIMEOUT)
    def serve(self):
        """The upstream process serves /ui and /v1/* on the same origin."""
        pass

    @modal.exit()
    def stop(self):
        terminate_server(getattr(self, "process", None))
        references_volume.commit()
        cache_volume.commit()
