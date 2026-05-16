"""
ComfyUI on Modal — Deploys ComfyUI with modular plugin/model management and an
optional OpenAI-compatible video generation API.

Architecture:
  - Modal builds a container with ComfyUI, custom nodes, and model weights.
  - A lightweight proxy (starlette) exposes port 8000. It routes /v1/video/*
    to the OpenAI-compatible video API and proxies all other traffic to the
    internal ComfyUI server on port 8001.
  - Models are downloaded at image build time and stored in a persistent
    Modal Volume (hf-hub-cache) so they survive container restarts.
  - GPU snapshots are enabled for fast cold starts after the first boot.

Config:
  - Copy models_example.py    → models.py    (model downloads)
  - Copy plugins_example.py   → plugins.py   (custom nodes + pip extras)
  - Edit workflows.py for video API endpoint mappings (optional)

Usage:
  modal serve comfyui/comfyui.py    # dev mode, logs in terminal
  modal run comfyui/comfyui.py      # trigger a test video generation
  modal deploy comfyui/comfyui.py   # production, persistent
"""

from __future__ import annotations

import json
import logging
import os
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
logger = logging.getLogger("comfyui")

# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

# Runtime
GPU_TYPE = "L40S"
N_GPU = 1
SCALEDOWN_WINDOW = 60           # idle seconds before container shutdown
TIMEOUT = 3600                  # max container lifetime (seconds)
STARTUP_TIMEOUT = 300           # seconds Modal waits for container to be ready
MAX_CONCURRENT_INPUTS = 10      # max concurrent requests per container

# Ports (ComfyUI runs internally, proxy runs exposed)
PROXY_PORT = 8000               # exposed to external traffic
COMFY_PORT = 8001               # internal, proxy forwards to this

# Paths
ROOT_DIR = Path(__file__).parent
COMFY_ROOT = Path("/root/comfy/ComfyUI")
COMFY_MODELS_ROOT = COMFY_ROOT / "models"

# Persistent storage
VOL_NAME = "hf-hub-cache"
VOL_MOUNT = "/cache"

# Features
ENABLE_MEMORY_SNAPSHOT = True
ENABLE_GPU_SNAPSHOT = True

# ═══════════════════════════════════════════════════════════════════════════════
# Import user configs (gracefully handle missing files at module level)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from models import MODELS_HF, MODELS_URL  # noqa: F811
except ImportError:
    logger.warning(
        "models.py not found — no models will be downloaded. "
        "Copy models_example.py to models.py and edit it."
    )
    MODELS_HF: list[dict] = []  # type: ignore[no-redef]
    MODELS_URL: list[dict] = []  # type: ignore[no-redef]

try:
    from plugins import PLUGINS_REGISTRY, PLUGINS_GIT, PLUGINS_PIP  # noqa: F811
except ImportError:
    logger.warning(
        "plugins.py not found — no custom nodes will be installed. "
        "Copy plugins_example.py to plugins.py and edit it."
    )
    PLUGINS_REGISTRY: list[str] = []  # type: ignore[no-redef]
    PLUGINS_GIT: list[dict] = []      # type: ignore[no-redef]
    PLUGINS_PIP: list[str] = []       # type: ignore[no-redef]

try:
    from workflows import WORKFLOWS  # noqa: F811
except ImportError:
    WORKFLOWS: dict = {}  # type: ignore[no-redef]


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_model_dir(model_dir: str) -> Path:
    """Resolve model_dir: absolute paths are used as-is, relative paths are
    placed under /root/comfy/ComfyUI/models/ (e.g. "checkpoints")."""
    p = Path(model_dir)
    return p if p.is_absolute() else COMFY_MODELS_ROOT / p


def hf_download(repo_id: str, filename: str, model_dir: str) -> None:
    """Download a model from Hugging Face Hub and symlink to target dir."""
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=VOL_MOUNT,
        token=os.environ.get("HF_TOKEN"),
    )
    _symlink_model(model_path, filename, model_dir)
    logger.info("hf: %s/%s -> %s", repo_id, filename, model_dir)


def url_download(url: str, filename: str, model_dir: str) -> None:
    """Download a model from a direct URL via aria2c and symlink to target dir."""
    import hashlib
    # Use URL hash to avoid filename collisions between different URLs
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:12]
    cache_dir = Path(VOL_MOUNT) / "ext" / url_hash
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / filename
    if not cached.exists():
        logger.info("downloading %s from %s ...", filename, url)
        subprocess.run(
            [
                "aria2c", "--console-log-level=error", "--summary-interval=0",
                "-x", "16", "-s", "16", "-o", filename, "-d", str(cache_dir), url,
            ],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    _symlink_model(str(cached), filename, model_dir)
    logger.info("url: %s -> %s", filename, model_dir)


def _symlink_model(source: str, filename: str, model_dir: str) -> None:
    """Create a symlink from the cached model to its ComfyUI model directory."""
    import shutil
    target_dir = resolve_model_dir(model_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / Path(filename).name
    if target.exists() or target.is_symlink():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.symlink_to(source)


def install_git_plugin(
    url: str, branch: str = "main", requirements: bool = False
) -> None:
    """Clone a custom node repo and optionally install its Python deps."""
    plugin_name = Path(url).stem
    dest = COMFY_ROOT / "custom_nodes" / plugin_name

    if dest.exists():
        logger.info("plugin already exists: %s", plugin_name)
        return

    logger.info("git clone %s (branch=%s) ...", plugin_name, branch)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", branch, url, str(dest)],
        check=True,
    )

    if requirements:
        req_file = dest / "requirements.txt"
        if req_file.exists():
            logger.info("pip install -r %s ...", req_file)
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
                check=True,
            )


def wait_for_port(port: int, timeout: int = 60) -> None:
    """Block until the port is accepting TCP connections."""
    import socket

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"port {port} not ready after {timeout}s")


def _log_gpu_info() -> None:
    """Log GPU hardware info for diagnostics."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,cuda_version",
                "--format=csv,noheader",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                logger.info("GPU: %s", line)
    except FileNotFoundError:
        logger.error("nvidia-smi not found — no GPU detected!")
    except Exception as e:
        logger.warning("could not query GPU info: %s", e)


# ═══════════════════════════════════════════════════════════════════════════════
# Build-time: model download (runs in Modal container during image build)
# ═══════════════════════════════════════════════════════════════════════════════

def download_all() -> None:
    """Download all configured models into the persistent volume."""
    logger.info(
        "downloading %d HF + %d URL models ...",
        len(MODELS_HF), len(MODELS_URL),
    )
    for m in MODELS_HF:
        hf_download(m["repo_id"], m["filename"], m["model_dir"])
    for m in MODELS_URL:
        url_download(m["url"], m["filename"], m["model_dir"])
    logger.info("all models downloaded")


# ═══════════════════════════════════════════════════════════════════════════════
# Persistent volume
# ═══════════════════════════════════════════════════════════════════════════════

vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Container image
# ═══════════════════════════════════════════════════════════════════════════════


def _hf_secrets() -> list[modal.Secret]:
    """Prefer Modal Secret 'huggingface-secret'; fall back to local HF_TOKEN."""
    try:
        s = modal.Secret.from_name("huggingface-secret")
        s.hydrate()
        return [s]
    except modal.exception.NotFoundError:
        token = os.environ.get("HF_TOKEN", "")
        if not token:
            logger.warning(
                "no huggingface-secret Modal Secret and no HF_TOKEN env — "
                "public models will download with throttled bandwidth"
            )
        return [modal.Secret.from_dict({"HF_TOKEN": token})]


image = modal.Image.debian_slim(python_version="3.12")

# Layer 1: base system + ComfyUI
image = (
    image
    .apt_install("git", "git-lfs", "libgl1-mesa-dev", "libglib2.0-0", "aria2")
    .pip_install_from_requirements(str(ROOT_DIR / "requirements_comfy.txt"))
    .run_commands("comfy --skip-prompt install --nvidia")
    .run_commands("git lfs install")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_XET_HIGH_PERFORMANCE": "1"})
)

# Layer 2: custom nodes (registry + git + pip)
if (ROOT_DIR / "plugins.py").exists():
    image = image.add_local_python_source("plugins", copy=True)

if PLUGINS_REGISTRY:
    image = image.run_commands(
        "comfy node install " + " ".join(PLUGINS_REGISTRY)
    )
    logger.info("registry plugins: %s", PLUGINS_REGISTRY)

if PLUGINS_GIT:
    image = image.run_function(_install_all_git_plugins)
    logger.info("git plugins: %d repos", len(PLUGINS_GIT))

if PLUGINS_PIP:
    image = image.pip_install(*PLUGINS_PIP)
    logger.info("pip extras: %s", PLUGINS_PIP)

# Workflow-based node installation (if workflow_api.json exists)
_wf_file = ROOT_DIR / "workflow_api.json"
if _wf_file.exists():
    image = image.add_local_file(
        _wf_file, "/root/workflow_api.json", copy=True
    ).run_commands("comfy node install-deps --workflow=/root/workflow_api.json")

# Add workflow config + JSONs into image (accessible by the API proxy)
if (ROOT_DIR / "workflows.py").exists():
    image = image.add_local_python_source("workflows", copy=True)

if (ROOT_DIR / "workflows").is_dir():
    image = image.add_local_dir(
        str(ROOT_DIR / "workflows"), "/root/workflows", copy=True
    )


def _install_all_git_plugins() -> None:
    """Build-time helper: install all git-based plugins."""
    for p in PLUGINS_GIT:
        install_git_plugin(
            url=p["url"],
            branch=p.get("branch", "main"),
            requirements=p.get("requirements", False),
        )


# Layer 3: download models into volume
if (ROOT_DIR / "models.py").exists():
    image = image.add_local_python_source("models", copy=True)

if MODELS_HF or MODELS_URL:
    image = image.run_function(
        download_all,
        volumes={VOL_MOUNT: vol},
        secrets=_hf_secrets(),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# App
# ═══════════════════════════════════════════════════════════════════════════════

app = modal.App("comfyui", image=image)


@app.cls(
    max_containers=1,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    volumes={VOL_MOUNT: vol},
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=TIMEOUT,
    enable_memory_snapshot=ENABLE_MEMORY_SNAPSHOT,
    experimental_options={"enable_gpu_snapshot": ENABLE_GPU_SNAPSHOT},
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_INPUTS)
class ComfyUI:
    """Modal class that runs ComfyUI behind a lightweight API proxy.

    Lifecycle:
      1. @modal.enter(snap=True)  start_checkpoint() — start ComfyUI + proxy,
         wait for ports, take GPU snapshot
      2. @modal.enter(snap=False) start_restore() — wait for ports (fast restore)
      3. @modal.web_server(8000)  proxy() — serve API + proxy to ComfyUI
      4. @modal.exit()            cleanup() — terminate processes
    """

    @modal.enter(snap=True)
    def start_checkpoint(self) -> None:
        """Cold start: launch ComfyUI + proxy, take GPU snapshot."""
        t_start = time.monotonic()
        logger.info("=== ComfyUI cold start ===")
        logger.info(
            "GPU: %s x %d | ComfyUI port: %d | Proxy port: %d",
            GPU_TYPE, N_GPU, COMFY_PORT, PROXY_PORT,
        )
        _log_gpu_info()

        self._start_comfy()
        self._start_proxy()

        elapsed = time.monotonic() - t_start
        logger.info("=== cold start complete in %.1fs ===", elapsed)

    @modal.enter(snap=False)
    def start_restore(self) -> None:
        """Warm start: restore from GPU snapshot."""
        logger.info("=== ComfyUI restore from snapshot ===")
        wait_for_port(COMFY_PORT, timeout=30)
        wait_for_port(PROXY_PORT, timeout=30)
        logger.info("=== restore complete ===")

    @modal.web_server(PROXY_PORT, startup_timeout=STARTUP_TIMEOUT)
    def proxy(self) -> None:
        """Placeholder — proxy runs as subprocess from start_checkpoint."""
        import time
        logger.info("=== proxy web_server active ===")
        while True:
            time.sleep(86400)

    @modal.exit()
    def cleanup(self) -> None:
        """Terminate ComfyUI and proxy subprocesses."""
        for attr in ("comfy_proc", "proxy_proc"):
            proc = getattr(self, attr, None)
            if proc is not None:
                try:
                    proc.terminate()
                except (ProcessLookupError, OSError):
                    pass
        logger.info("=== cleanup complete ===")

    # ── Subprocess management ────────────────────────────────────────────

    def _start_comfy(self) -> None:
        """Launch ComfyUI server on the internal port."""
        logger.info("starting ComfyUI on port %d ...", COMFY_PORT)
        self.comfy_proc = subprocess.Popen(
            f"comfy launch -- --listen 0.0.0.0 --port {COMFY_PORT}",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        logger.info("ComfyUI PID: %d", self.comfy_proc.pid)
        _stream_output(self.comfy_proc, "COMFY")
        wait_for_port(COMFY_PORT, timeout=300)
        logger.info("ComfyUI ready on port %d", COMFY_PORT)

    def _start_proxy(self) -> None:
        """Launch the proxy + API server on the exposed port."""
        logger.info("starting proxy on port %d ...", PROXY_PORT)
        self.proxy_proc = subprocess.Popen(
            [
                sys.executable, "-u", "-c", _PROXY_SERVER_SCRIPT,
            ],
            env={"COMFY_PORT": str(COMFY_PORT), **os.environ},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        logger.info("proxy PID: %d", self.proxy_proc.pid)
        _stream_output(self.proxy_proc, "PROXY")
        wait_for_port(PROXY_PORT, timeout=30)
        logger.info("proxy ready on port %d", PROXY_PORT)


# ═══════════════════════════════════════════════════════════════════════════════
# Local entrypoint (testing)
# ═══════════════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
async def main(
    prompt: str = "A cinematic drone shot of a mountain valley at golden hour",
    model: str = "ltx2.3",
):
    """Test the deployed server by sending a video generation request.

    Usage:
      modal run comfyui/comfyui.py
      modal run comfyui/comfyui.py --prompt "A cat walking in the rain"
    """
    import httpx

    url = await ComfyUI.proxy.get_web_url.aio()

    async with httpx.AsyncClient(base_url=str(url), timeout=600.0) as client:
        logger.info("submitting video generation: %s", prompt)
        resp = await client.post(
            "/v1/video/generations",
            json={"model": model, "prompt": prompt},
        )
        result = resp.json()
        logger.info("result: %s", json.dumps(result, indent=2))


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _stream_output(proc: subprocess.Popen, tag: str) -> None:
    """Stream subprocess output to structured logs in a daemon thread."""
    import threading

    def _reader() -> None:
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            if any(kw in line.lower() for kw in ("error", "exception", "traceback")):
                logger.error("%s: %s", tag, line)
            elif any(kw in line.lower() for kw in ("loaded", "ready", "serving", "running", "uvicorn")):
                logger.info("%s: %s", tag, line)
            elif any(kw in line.lower() for kw in ("download", "loading", "starting")):
                logger.info("%s: %s", tag, line)
            else:
                logger.debug("%s: %s", tag, line)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()


# ═══════════════════════════════════════════════════════════════════════════════
# Proxy + Video API server (runs as subprocess in the container)
# ═══════════════════════════════════════════════════════════════════════════════

_PROXY_SERVER_SCRIPT = r'''
import asyncio, json, logging, os, sys, time, uuid
from collections import OrderedDict
from copy import deepcopy

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse
from starlette.routing import Route

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] proxy: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("proxy")

COMFY_PORT = int(os.environ.get("COMFY_PORT", "8001"))
COMFY_URL = f"http://127.0.0.1:{COMFY_PORT}"
WF_DIR = "/root/workflows"


# ── Workflow helpers ──────────────────────────────────────────────────────

def _load_wf(filepath):
    path = os.path.join(WF_DIR, filepath)
    with open(path) as f:
        return json.load(f)


def _ui_to_api(ui_workflow):
    """Convert a ComfyUI UI-exported workflow to the /prompt API format.

    UI format:
      {"nodes": [{"id": N, "type": "...", "inputs": [...], "widgets_values": [...]}],
       "links": [[lid, src, src_slot, dst, dst_slot, type]]}

    API format:
      {"N": {"class_type": "...", "inputs": {"param": value_or_link}},
       ...}
    """
    nodes = {n["id"]: n for n in ui_workflow["nodes"]}
    links = ui_workflow.get("links", [])

    # Build a lookup: (dst_id, dst_slot) -> (src_id, src_slot)
    link_map = {}
    for link in links:
        lid, src_id, src_slot, dst_id, dst_slot, _ = link
        link_map[(dst_id, dst_slot)] = (src_id, src_slot)

    api_prompt = OrderedDict()
    for node in ui_workflow["nodes"]:
        entry = {"class_type": node["type"], "inputs": OrderedDict()}
        node_inputs = node.get("inputs", [])

        for idx, inp in enumerate(node_inputs):
            name = inp["name"]
            slot = idx

            if (node["id"], slot) in link_map:
                src_id, src_slot = link_map[(node["id"], slot)]
                entry["inputs"][name] = [str(src_id), src_slot]
            elif "widget" in inp and inp["widget"] is not None:
                entry["inputs"][name] = inp["widget"].get("value", None)

        # Fallback: use widgets_values for nodes without input definitions
        if not node_inputs and node.get("widgets_values"):
            wv = node["widgets_values"]
            # Try to match widget names from a known mapping
            if node["type"] == "CLIPTextEncode":
                if wv:
                    entry["inputs"]["text"] = wv[0]
            elif node["type"] in ("PrimitiveBoolean",):
                if wv:
                    entry["inputs"]["boolean"] = wv[0]

        api_prompt[str(node["id"])] = entry

    return api_prompt


def _inject_params(ui_wf, params, param_map):
    """Inject user params into a UI-format workflow's widgets_values."""
    for api_key, target in param_map.items():
        if api_key not in params:
            continue
        value = params[api_key]
        parts = target.split(".")
        node_id = int(parts[0])
        field_type = parts[1]
        field_idx = int(parts[2]) if len(parts) > 2 else 0

        for node in ui_wf.get("nodes", []):
            if node["id"] == node_id:
                if field_type == "widgets":
                    wv = node.setdefault("widgets_values", [])
                    while len(wv) <= field_idx:
                        wv.append(None)
                    wv[field_idx] = value
                break


def _extract_files(outputs):
    """Extract generated file info from ComfyUI history outputs."""
    files = []
    for node_id, output in outputs.items():
        for media_type in ("gifs", "images", "videos"):
            for item in output.get(media_type, []):
                files.append({
                    "filename": item.get("filename", ""),
                    "subfolder": item.get("subfolder", ""),
                    "type": item.get("type", media_type.rstrip("s")),
                })
    return files


# ── Routes ─────────────────────────────────────────────────────────────────

async def create_video(request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    model_id = body.get("model", "ltx2.3")
    from workflows import WORKFLOWS
    wf_config = WORKFLOWS.get(model_id)
    if not wf_config:
        return JSONResponse({
            "error": f"model {model_id!r} not found",
            "available": list(WORKFLOWS.keys()),
        }, status_code=404)

    ui_wf = _load_wf(wf_config["file"])
    _inject_params(ui_wf, body, wf_config.get("param_map", {}))
    api_prompt = _ui_to_api(ui_wf)

    client_id = str(uuid.uuid4())
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{COMFY_URL}/prompt",
            json={"prompt": api_prompt, "client_id": client_id},
        )
        if resp.status_code != 200:
            body = resp.text[:500]
            return JSONResponse(
                {"error": f"ComfyUI rejected prompt ({resp.status_code})", "detail": body},
                status_code=502,
            )
        result = resp.json()

    prompt_id = result.get("prompt_id")
    if not prompt_id:
        return JSONResponse({"error": "no prompt_id from ComfyUI"}, status_code=502)

    # Validate and clamp timeout
    raw_timeout = body.get("timeout", 600)
    try:
        timeout = max(10, min(int(raw_timeout), 3600))
    except (TypeError, ValueError):
        timeout = 600

    video = await _poll(prompt_id, timeout=timeout)

    files = video.get("files", [])
    for f in files:
        subfolder = f.get("subfolder", "")
        filename = f["filename"]
        f["url"] = f"/view?filename={filename}&subfolder={subfolder}&type=output"

    return JSONResponse({
        "id": prompt_id,
        "status": video["status"],
        "output": files,
    })


async def get_video(request):
    prompt_id = request.path_params.get("generation_id")
    video = await _poll(prompt_id, timeout=5)
    files = video.get("files", [])
    for f in files:
        subfolder = f.get("subfolder", "")
        filename = f["filename"]
        f["url"] = f"/view?filename={filename}&subfolder={subfolder}&type=output"
    return JSONResponse({
        "id": prompt_id,
        "status": video["status"],
        "output": files,
    })


async def _poll(prompt_id, timeout=600):
    deadline = time.time() + timeout
    client = httpx.AsyncClient(timeout=10.0)
    try:
        while time.time() < deadline:
            resp = await client.get(f"{COMFY_URL}/history/{prompt_id}")
            if resp.status_code == 200:
                data = resp.json()
                if prompt_id in data:
                    entry = data[prompt_id]
                    # Check for ComfyUI errors
                    status_meta = entry.get("status", {})
                    if status_meta.get("status") == "error":
                        error_msg = status_meta.get("messages", ["unknown error"])[0]
                        return {"status": "failed", "error": error_msg, "files": []}
                    outputs = entry.get("outputs", {})
                    files = _extract_files(outputs)
                    if files:
                        return {"status": "completed", "files": files}
            await asyncio.sleep(2)
    finally:
        await client.aclose()
    return {"status": "processing", "files": []}


async def proxy_all(request):
    path = request.url.path
    qs = str(request.url.query)
    url = f"{COMFY_URL}{path}"
    if qs:
        url = url + "?" + qs

    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host",)
    }
    body = await request.body()

    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.request(
            request.method, url, headers=headers, content=body,
        )

    r_headers = {
        k: v for k, v in resp.headers.items()
        if k.lower() not in ("transfer-encoding", "content-length")
    }

    return StreamingResponse(
        resp.aiter_bytes(),
        status_code=resp.status_code,
        headers=r_headers,
        media_type=resp.headers.get("content-type"),
    )


# ── App ──────────────────────────────────────────────────────────────────────

app = Starlette(routes=[
    Route("/v1/video/generations", create_video, methods=["POST"]),
    Route("/v1/video/generations/{generation_id:str}", get_video, methods=["GET"]),
    Route("/health", lambda _: JSONResponse({"status": "ok"}), methods=["GET"]),
    Route("/{path:path}", proxy_all, methods=[
        "GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD",
    ]),
])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
'''
