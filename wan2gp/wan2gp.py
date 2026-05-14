"""
Wan2GP on Modal — Deploys the Wan2GP video generation Gradio app to Modal.

Architecture:
  - Modal builds a container image with PyTorch, xformers, and Wan2GP deps.
  - At container startup, data dirs (ckpts, loras, outputs) are symlinked into
    a persistent Modal Volume so checkpoints and outputs survive restarts.
  - wgp.py is launched as a subprocess with Gradio's --share flag, which
    creates a public tunnel URL (*.gradio.live) that bypasses Modal's HTTP
    proxy. This avoids Content-Length mismatch errors that occur when Gradio
    streams responses (SSE, progressive loading) through Modal's reverse proxy.
  - Modal's @modal.web_server is still used to keep the container alive and
    provide a fallback URL, but the primary access is via the Gradio share URL
    printed in the logs.

Usage:
  modal serve wan2gp/wan2gp.py   # dev mode, logs in terminal
  modal deploy wan2gp/wan2gp.py  # production, persistent
"""

import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("wan2gp")

# ── Runtime config ──────────────────────────────────────────────────────────
# Available GPU types: T4, L4, A10, L40S, A100, A100-40GB, A100-80GB,
# RTX-PRO-6000, H100, H100!, H200, B200, B200+
# See https://modal.com/pricing for details.
GPU_TYPE = "A100-80GB"
TIMEOUT = 3600  # max seconds a container stays alive
MAX_CONCURRENT_INPUTS = 3  # max concurrent requests per container
GRADIO_PORT = 7860  # port Gradio listens on inside the container
WAN2GP_PROFILE = "1"  # offloading profile: 1 = high VRAM
STARTUP_TIMEOUT = 300  # seconds Modal waits for the container to be ready
GRADIO_SHARE = True  # create a public *.gradio.live tunnel URL

# ── Paths ───────────────────────────────────────────────────────────────────
WAN2GP_ROOT = "/root/Wan2GP"  # cloned repo location inside container
VOL_MOUNT = "/mnt/wan2gp-data"  # where the persistent volume is mounted
VOL_NAME = "wan2gp-data"  # Modal Volume name (created if missing)

# ── Dependency versions ─────────────────────────────────────────────────────
PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
TORCH_VERSION = "2.8.0"
TORCHVISION_VERSION = "0.23.0"
TORCHAUDIO_VERSION = "2.8.0"
XFORMERS_VERSION = "0.0.32.post2"

# ── Container image ──────────────────────────────────────────────────────────
# Built once and cached by Modal. Each .run_commands / .pip_install creates a
# new image layer that is only rebuilt when its inputs change.
wan2gp_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libglib2.0-0", "libgl1", "libportaudio2")
    .run_commands(f"git clone https://github.com/deepbeepmeep/Wan2GP.git {WAN2GP_ROOT}")
    .workdir(WAN2GP_ROOT)
    # Install Wan2GP's own requirements first (includes mmgp, gradio, etc.)
    .run_commands("pip install -r /root/Wan2GP/requirements.txt")
    # Install PyTorch with CUDA 12.8 from the PyTorch wheel index
    .pip_install(
        f"torch=={TORCH_VERSION}",
        f"torchvision=={TORCHVISION_VERSION}",
        f"torchaudio=={TORCHAUDIO_VERSION}",
        index_url=PYTORCH_INDEX_URL,
    )
    # xformers must match the PyTorch + CUDA version
    .pip_install(f"xformers=={XFORMERS_VERSION}", index_url=PYTORCH_INDEX_URL)
    # onnxruntime-gpu for whisper/audio models
    .pip_install(
        "onnxruntime-gpu",
        extra_index_url="https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/",
    )
    # Pin numpy to a range compatible with all Wan2GP dependencies
    .pip_install("numpy>=1.24,<2.3")
    # ── Upstream patches ──────────────────────────────────────────────────
    # Patch 1: Headless servers don't have TkAgg; use the Agg backend instead.
    .run_commands(
        f"sed -i \"s/matplotlib.use('TkAgg')/matplotlib.use('Agg')/\" "
        f"{WAN2GP_ROOT}/preprocessing/matanyone/tools/interact_tools.py || true",
    )
)

# ── Persistent storage ─────────────────────────────────────────────────────
# Modal Volumes persist data across container restarts. Model checkpoints,
# LoRA weights, and generated outputs are stored here so they don't need to be
# re-downloaded on every cold start.
wan2gp_volume = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

app = modal.App("wan2gp")


@app.cls(
    image=wan2gp_image,
    gpu=GPU_TYPE,
    volumes={VOL_MOUNT: wan2gp_volume},
    timeout=TIMEOUT,
    min_containers=1,  # always keep 1 container running so Gradio share URL stays alive
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_INPUTS)
class Wan2GP:
    """Modal class that runs Wan2GP's Gradio interface.

    Lifecycle:
      1. @modal.enter() setup() — prepare volumes, env vars
      2. @modal.web_server() launch() — start wgp.py as a subprocess
      3. User accesses the app via the Gradio share URL (*.gradio.live)
         printed in the container logs (NOT the Modal URL)
    """

    @modal.enter()
    def setup(self):
        """Run once when a new container starts. Prepares the environment."""
        t_start = time.monotonic()
        logger.info("=== Wan2GP container startup ===")
        logger.info(
            "PyTorch %s | torchvision %s | torchaudio %s | xformers %s",
            TORCH_VERSION,
            TORCHVISION_VERSION,
            TORCHAUDIO_VERSION,
            XFORMERS_VERSION,
        )
        logger.info(
            "GPU: %s | Timeout: %ds | Max concurrent: %d | Profile: %s",
            GPU_TYPE,
            TIMEOUT,
            MAX_CONCURRENT_INPUTS,
            WAN2GP_PROFILE,
        )

        self._log_gpu_info()
        self._prepare_volume_dirs()
        self._link_repo_to_volume()
        self._set_cache_env()
        wan2gp_volume.commit()
        logger.info("Volume committed successfully")

        # Disable Gradio telemetry
        os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

        elapsed = time.monotonic() - t_start
        logger.info("=== Setup complete in %.1fs ===", elapsed)

    @modal.web_server(GRADIO_PORT, startup_timeout=STARTUP_TIMEOUT)
    def launch(self):
        """Start wgp.py as a subprocess with Gradio's --share flag.

        Gradio's --share creates a public tunnel URL (*.gradio.live) that
        connects directly to the Gradio server without going through Modal's
        HTTP reverse proxy. This avoids Content-Length mismatch errors that
        occur when Gradio streams responses (SSE, progressive loading, file
        uploads) through the proxy.

        The Modal URL still works as a fallback, but may encounter the
        proxy-related errors. Always use the Gradio share URL for the best
        experience. The URL is printed in the container logs after startup.
        """
        logger.info(
            "=== Launching Wan2GP Gradio on port %d (startup_timeout=%ds) ===",
            GRADIO_PORT,
            STARTUP_TIMEOUT,
        )
        logger.info("Profile: %s | CWD: %s", WAN2GP_PROFILE, WAN2GP_ROOT)

        cmd = [
            sys.executable,
            "-u",  # unbuffered stdout/stderr for real-time logging
            "wgp.py",
            "--listen",  # bind to 0.0.0.0
            "--server-port",
            str(GRADIO_PORT),
            "--profile",
            WAN2GP_PROFILE,
        ]
        # --share creates a public *.gradio.live tunnel URL that bypasses
        # Modal's HTTP proxy, avoiding Content-Length mismatch on streaming
        if GRADIO_SHARE:
            cmd.append("--share")

        logger.info("Command: %s", " ".join(cmd))

        proc = subprocess.Popen(
            cmd,
            cwd=WAN2GP_ROOT,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # line-buffered for real-time log streaming
        )
        logger.info("wgp.py PID: %d", proc.pid)

        # Stream wgp.py output to structured logs so Gradio share URL,
        # errors, and model loading progress are visible in Modal logs
        self._stream_wgp_output(proc)

    def _stream_wgp_output(self, proc):
        """Stream subprocess output to structured logs in a daemon thread.

        Routes lines to the appropriate log level:
          - INFO: Gradio URLs, model downloads/loading progress
          - ERROR: tracebacks, exceptions, errors
          - DEBUG: everything else (routine Gradio noise)
        """
        import threading

        def _reader():
            for line in proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                # Gradio share URL and server startup messages
                if "Running on" in line or "gradio.live" in line:
                    logger.info("GRADIO: %s", line)
                # Error-class lines that need attention
                elif any(
                    kw in line.lower() for kw in ("error", "exception", "traceback")
                ):
                    logger.error("WGP: %s", line)
                # Useful progress info (model download, loading, etc.)
                elif any(kw in line.lower() for kw in ("download", "loading", "model")):
                    logger.info("WGP: %s", line)
                else:
                    logger.debug("WGP: %s", line)

        t = threading.Thread(target=_reader, daemon=True)
        t.start()

    # ── Volume & environment helpers ──────────────────────────────────────

    def _prepare_volume_dirs(self):
        """Create the directory structure inside the persistent volume."""
        wan_data_root = Path(VOL_MOUNT)
        dirs = {
            "data_root": wan_data_root,
            "ckpts": wan_data_root / "ckpts",
            "loras": wan_data_root / "loras",
            "outputs": wan_data_root / "outputs",
            "cache": wan_data_root / "cache",
            "ltx2_loras": wan_data_root / "loras" / "ltx2",
            "ltx2_22b_loras": wan_data_root / "loras" / "ltx2_22B",
        }
        for label, d in dirs.items():
            d.mkdir(parents=True, exist_ok=True)
            logger.info("Volume dir [%s]: %s", label, d)

    def _link_repo_to_volume(self):
        """Symlink the repo's ckpts/loras/outputs dirs to the persistent volume
        so downloaded models and generated files survive container restarts."""
        wan_data_root = Path(VOL_MOUNT)
        links = {
            "ckpts": wan_data_root / "ckpts",
            "loras": wan_data_root / "loras",
            "outputs": wan_data_root / "outputs",
        }
        for repo_subdir, vol_dir in links.items():
            self._merge_and_link(Path(WAN2GP_ROOT) / repo_subdir, vol_dir)

    def _set_cache_env(self):
        """Point all ML framework cache dirs into the persistent volume so
        downloaded model weights and tokenizer caches survive restarts."""
        wan_cache_dir = Path(VOL_MOUNT) / "cache"
        cache_env = {
            "WAN_CACHE_DIR": str(wan_cache_dir),
            "HF_HOME": str(wan_cache_dir / "huggingface"),
            "HUGGINGFACE_HUB_CACHE": str(wan_cache_dir / "huggingface" / "hub"),
            "TRANSFORMERS_CACHE": str(wan_cache_dir / "huggingface" / "transformers"),
            "TORCH_HOME": str(wan_cache_dir / "torch"),
            "XDG_CACHE_HOME": str(wan_cache_dir / ".cache"),
        }
        self.env = os.environ.copy()
        self.env.update(cache_env)
        # Also set on os.environ for any in-process code that reads env vars
        os.environ.update(cache_env)
        for k, v in cache_env.items():
            logger.info("Env %s=%s", k, v)

    # ── Diagnostics ───────────────────────────────────────────────────────

    @staticmethod
    def _log_gpu_info():
        """Log GPU hardware info from nvidia-smi and torch.cuda for debugging."""
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
            else:
                logger.warning(
                    "nvidia-smi query failed (rc=%d): %s",
                    result.returncode,
                    result.stderr.strip(),
                )
        except FileNotFoundError:
            logger.error("nvidia-smi not found - no GPU detected!")
        except Exception as e:
            logger.warning("Could not query GPU info: %s", e)

        try:
            import torch

            logger.info("torch.cuda.is_available: %s", torch.cuda.is_available())
            if torch.cuda.is_available():
                logger.info("torch.cuda.device_count: %d", torch.cuda.device_count())
                logger.info(
                    "torch.cuda.current_device: %d", torch.cuda.current_device()
                )
                logger.info(
                    "torch.cuda.get_device_name: %s", torch.cuda.get_device_name(0)
                )
                free, total = torch.cuda.mem_get_info(0)
                logger.info(
                    "GPU memory: %.1f GB free / %.1f GB total", free / 1e9, total / 1e9
                )
        except ImportError:
            logger.warning("torch not importable at setup time")

    # ── Symlink helper ────────────────────────────────────────────────────

    @staticmethod
    def _merge_and_link(repo_path: Path, vol_dir: Path):
        """Replace a repo subdirectory with a symlink to the persistent volume.

        If the repo directory already has files (e.g. from a fresh clone), they
        are moved into the volume first to preserve any default files. If the
        symlink already points to the correct volume dir, this is a no-op.
        """
        vol_dir.mkdir(parents=True, exist_ok=True)
        if repo_path.is_symlink():
            target = repo_path.resolve()
            if target != vol_dir.resolve():
                logger.warning(
                    "Symlink %s -> %s exists but expected %s; skipping",
                    repo_path,
                    target,
                    vol_dir,
                )
            else:
                logger.info("Symlink already OK: %s -> %s", repo_path.name, vol_dir)
            return
        if repo_path.exists() and repo_path.is_dir():
            # Move any existing files from the repo dir into the volume
            file_count = sum(1 for _ in repo_path.iterdir())
            if file_count > 0:
                logger.info(
                    "Merging %d items from %s into %s", file_count, repo_path, vol_dir
                )
                for child in list(repo_path.iterdir()):
                    dst = vol_dir / child.name
                    if dst.exists():
                        logger.debug("Skip existing: %s", dst)
                        continue
                    shutil.move(str(child), str(dst))
            else:
                logger.info("Dir %s is empty, removing for symlink", repo_path)
            repo_path.rmdir()
        else:
            logger.info("Dir %s does not exist, creating parent for symlink", repo_path)
            repo_path.parent.mkdir(parents=True, exist_ok=True)
        repo_path.symlink_to(vol_dir, target_is_directory=True)
        logger.info("Linked %s -> %s", repo_path, vol_dir)
