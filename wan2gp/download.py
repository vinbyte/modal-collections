"""
Utility script to pre-download Wan2GP models to the Modal volume using a CPU-only instance.
This avoids paying for expensive GPU time just to download model weights.

Usage:
  modal run wan2gp/download.py
  modal run wan2gp/download.py --model-name minimax_h3_fl2va
"""

import logging
import os
import sys
import json
import subprocess
from pathlib import Path
import modal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("wan2gp-download")

app = modal.App("wan2gp-download")
VOL_MOUNT = "/mnt/wan2gp-data"
VOL_NAME = "wan2gp-data"
WAN2GP_ROOT = "/root/Wan2GP"

wan2gp_volume = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
wan2gp_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "ffmpeg")
    .run_commands("git clone https://github.com/deepbeepmeep/Wan2GP.git /root/Wan2GP")
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "torchaudio==2.8.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("xformers==0.0.32.post2", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("hf_transfer")
    .run_commands("pip install -r /root/Wan2GP/requirements.txt")
)

@app.function(
    image=wan2gp_image,
    volumes={VOL_MOUNT: wan2gp_volume},
    timeout=7200,
    cpu=4.0,
    memory=8192,
)
def download_model(model_name: str = "minimax_h3_fl2va"):
    """
    Downloads model weights to the persistent volume using CPU.
    
    Usage: modal run wan2gp/download.py --model-name minimax_h3_fl2va
    """
    logger.info("Downloading models for %s on CPU...", model_name)
    wan_cache_dir = Path(VOL_MOUNT) / "cache"
    env = os.environ.copy()
    env.update({
        "WAN_CACHE_DIR": str(wan_cache_dir),
        "HF_HOME": str(wan_cache_dir / "huggingface"),
        "HUGGINGFACE_HUB_CACHE": str(wan_cache_dir / "huggingface" / "hub"),
        "TRANSFORMERS_CACHE": str(wan_cache_dir / "huggingface" / "transformers"),
        "TORCH_HOME": str(wan_cache_dir / "torch"),
        "XDG_CACHE_HOME": str(wan_cache_dir / ".cache"),
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
    
    # 1. Mock torch.cuda so wgp.py doesn't crash on CPU
    wrapper_path = os.path.join(WAN2GP_ROOT, "cpu_mock.py")
    with open(wrapper_path, "w") as f:
        f.write('''\
import sys
import runpy
import torch

if not hasattr(torch, "cuda") or not torch.cuda.is_available():
    if not hasattr(torch, "cuda"):
        import types
        torch.cuda = types.ModuleType("cuda")
    
    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = lambda: 1
    torch.cuda.current_device = lambda: 0
    torch.cuda.get_device_name = lambda *a: "Mock GPU"
    torch.cuda.get_device_capability = lambda *a: (8, 0)
    torch.cuda.empty_cache = lambda: None
    torch.cuda.is_bf16_supported = lambda: True
    torch.cuda.get_arch_list = lambda: ["sm_80"]
    torch.cuda._is_compiled = lambda: True
    torch.cuda.OutOfMemoryError = Exception
    
    class Props:
        total_memory = 80 * 1024 * 1024 * 1024
        major = 8
        minor = 0
    torch.cuda.get_device_properties = lambda *a: Props()
    torch.cuda.is_initialized = lambda: True

sys.argv = ["wgp.py", "--test", "--gpu", "cpu"]
runpy.run_path("wgp.py", run_name="__main__")
''')

    # 3. Execute
    # 2. Run wgp.py until it finishes initializing and generates default wgp_config.json
    logger.info("Initializing Wan2GP to generate default config (this may take a few minutes)...")
    cmd = [sys.executable, "-u", "cpu_mock.py"]
    try:
        init_proc = subprocess.Popen(
            cmd, cwd=WAN2GP_ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in init_proc.stdout:
            if line.strip():
                logger.info("INIT: %s", line.strip())
            if "Running on local URL" in line:
                logger.info("Default config generated! Terminating init process...")
                init_proc.terminate()
                init_proc.wait(timeout=5)
                break
    except Exception as e:
        logger.error("Error during init process: %s", e)
        if init_proc:
            init_proc.kill()

    # 3. Update the generated config to preload our specific model
    logger.info("Patching config for %s...", model_name)
    config_path = os.path.join(WAN2GP_ROOT, "wgp_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {}
    
    config["preload_model_policy"] = ["P"]
    config["last_model_type"] = model_name
    
    with open(config_path, "w") as f:
        json.dump(config, f)

    # 4. Run wgp.py again to actually trigger the download
    logger.info("Starting model download process...")
    success_terminated = False
    proc = subprocess.Popen(
        cmd, cwd=WAN2GP_ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        line = line.rstrip()
        if not line: continue
        if any(kw in line.lower() for kw in ("error", "exception", "traceback")):
            logger.error("WGP: %s", line)
        else:
            logger.info("WGP: %s", line)
        
        # Once Gradio starts, the model is successfully downloaded and loaded into memory
        if "Running on local URL" in line:
            logger.info("Model download and loading complete! Terminating process...")
            success_terminated = True
            proc.terminate()
            break
            
        # Or, if it crashes on mmgp's CUDA stream init, it means downloads are finished!
        if "CUDA driver version is insufficient" in line or "AcceleratorError" in line:
            logger.info("Model downloaded! Hit expected CUDA driver error on CPU offloader. Terminating...")
            success_terminated = True
            proc.terminate()
            break
            
    proc.wait(timeout=7200)
    if not success_terminated and proc.returncode != 0:
        raise RuntimeError(f"Model download failed with exit code {proc.returncode}")
    logger.info("Model downloaded successfully")

