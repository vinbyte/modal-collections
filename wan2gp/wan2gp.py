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
# Available option for GPU_TYPE. Please refer to https://modal.com/pricing for the pricing details of each GPU type.
# T4
# L4
# A10
# L40S
# A100
# A100-40GB
# A100-80GB
# RTX-PRO-6000
# H100/H100!
# H200
# B200/B200+
GPU_TYPE = "A100-80GB"
TIMEOUT = 3600
MAX_CONCURRENT_INPUTS = 10
GRADIO_PORT = 7860
WAN2GP_PROFILE = "1"
STARTUP_TIMEOUT = 300

# ── Paths ───────────────────────────────────────────────────────────────────
WAN2GP_ROOT = "/root/Wan2GP"
VOL_MOUNT = "/mnt/wan2gp-data"
VOL_NAME = "wan2gp-data"

# ── Dependency versions ─────────────────────────────────────────────────────
PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
TORCH_VERSION = "2.8.0"
TORCHVISION_VERSION = "0.23.0"
TORCHAUDIO_VERSION = "2.8.0"
XFORMERS_VERSION = "0.0.32.post2"

# ── Image ────────────────────────────────────────────────────────────────────
wan2gp_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libglib2.0-0", "libgl1", "libportaudio2")
    .run_commands(f"git clone https://github.com/deepbeepmeep/Wan2GP.git {WAN2GP_ROOT}")
    .workdir(WAN2GP_ROOT)
    .run_commands("pip install -r /root/Wan2GP/requirements.txt")
    .pip_install(
        f"torch=={TORCH_VERSION}",
        f"torchvision=={TORCHVISION_VERSION}",
        f"torchaudio=={TORCHAUDIO_VERSION}",
        index_url=PYTORCH_INDEX_URL,
    )
    .pip_install(f"xformers=={XFORMERS_VERSION}", index_url=PYTORCH_INDEX_URL)
    .pip_install(
        "onnxruntime-gpu",
        extra_index_url="https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/",
    )
    .pip_install("numpy>=1.24,<2.3")
    .run_commands(
        f"sed -i \"s/matplotlib.use('TkAgg')/matplotlib.use('Agg')/\" {WAN2GP_ROOT}/preprocessing/matanyone/tools/interact_tools.py || true"
    )
)

wan2gp_volume = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

app = modal.App("wan2gp")


@app.cls(
    image=wan2gp_image,
    gpu=GPU_TYPE,
    volumes={VOL_MOUNT: wan2gp_volume},
    timeout=TIMEOUT,
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_INPUTS)
class Wan2GP:
    @modal.enter()
    def setup(self):
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

        wan_data_root = Path(VOL_MOUNT)
        wan_ckpts_dir = wan_data_root / "ckpts"
        wan_loras_dir = wan_data_root / "loras"
        wan_outputs_dir = wan_data_root / "outputs"
        wan_cache_dir = wan_data_root / "cache"
        wan_ltx2_loras_dir = wan_loras_dir / "ltx2"
        wan_ltx2_22b_loras_dir = wan_loras_dir / "ltx2_22B"

        dirs_to_create = {
            "data_root": wan_data_root,
            "ckpts": wan_ckpts_dir,
            "loras": wan_loras_dir,
            "outputs": wan_outputs_dir,
            "cache": wan_cache_dir,
            "ltx2_loras": wan_ltx2_loras_dir,
            "ltx2_22b_loras": wan_ltx2_22b_loras_dir,
        }
        for label, d in dirs_to_create.items():
            d.mkdir(parents=True, exist_ok=True)
            logger.info("Volume dir [%s]: %s", label, d)

        wan2gp_root = Path(WAN2GP_ROOT)

        links = {
            "ckpts": wan_ckpts_dir,
            "loras": wan_loras_dir,
            "outputs": wan_outputs_dir,
        }
        for repo_subdir, vol_dir in links.items():
            self._merge_and_link(wan2gp_root / repo_subdir, vol_dir)

        self.env = os.environ.copy()
        cache_env = {
            "WAN_CACHE_DIR": str(wan_cache_dir),
            "HF_HOME": str(wan_cache_dir / "huggingface"),
            "HUGGINGFACE_HUB_CACHE": str(wan_cache_dir / "huggingface" / "hub"),
            "TRANSFORMERS_CACHE": str(wan_cache_dir / "huggingface" / "transformers"),
            "TORCH_HOME": str(wan_cache_dir / "torch"),
            "XDG_CACHE_HOME": str(wan_cache_dir / ".cache"),
        }
        self.env.update(cache_env)
        for k, v in cache_env.items():
            logger.info("Env %s=%s", k, v)

        wan2gp_volume.commit()
        logger.info("Volume committed successfully")

        elapsed = time.monotonic() - t_start
        logger.info("=== Setup complete in %.1fs ===", elapsed)

    @modal.web_server(GRADIO_PORT, startup_timeout=STARTUP_TIMEOUT)
    def launch(self):
        logger.info(
            "=== Launching Wan2GP Gradio on port %d (startup_timeout=%ds) ===",
            GRADIO_PORT,
            STARTUP_TIMEOUT,
        )
        logger.info("Profile: %s | CWD: %s", WAN2GP_PROFILE, WAN2GP_ROOT)

        cmd = [
            sys.executable,
            "-u",
            "wgp.py",
            "--listen",
            "--server-port",
            str(GRADIO_PORT),
            "--profile",
            WAN2GP_PROFILE,
        ]
        logger.info("Command: %s", " ".join(cmd))

        proc = subprocess.Popen(
            cmd,
            cwd=WAN2GP_ROOT,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        logger.info("wgp.py PID: %d", proc.pid)

        def _stream_output():
            for line in proc.stdout:
                line = line.rstrip()
                if "Running on" in line or "http" in line.lower():
                    logger.info("GRADIO: %s", line)
                elif (
                    "error" in line.lower()
                    or "exception" in line.lower()
                    or "traceback" in line.lower()
                ):
                    logger.error("WGP: %s", line)
                elif (
                    "download" in line.lower()
                    or "loading" in line.lower()
                    or "model" in line.lower()
                ):
                    logger.info("WGP: %s", line)
                else:
                    logger.debug("WGP: %s", line)

        import threading

        t = threading.Thread(target=_stream_output, daemon=True)
        t.start()

    @staticmethod
    def _log_gpu_info():
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

    @staticmethod
    def _merge_and_link(repo_path: Path, vol_dir: Path):
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
            file_count = sum(1 for _ in repo_path.iterdir())
            if file_count > 0:
                logger.info(
                    "Merging %d items from %s into %s", file_count, repo_path, vol_dir
                )
                for child in list(repo_path.iterdir()):
                    dst = vol_dir / child.name
                    if dst.exists():
                        if child.is_dir() and dst.is_dir():
                            logger.debug("Skip existing dir: %s", dst)
                            continue
                        logger.debug("Skip existing file: %s", dst)
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
