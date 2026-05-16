"""
Custom nodes and extra packages to install during ComfyUI image build.

Three plugin types are supported:

  1. PLUGINS_REGISTRY — comfy-cli registry node IDs.
     Install command: comfy node install <id>

  2. PLUGINS_GIT — git repositories (custom nodes not on registry).
     Fields: url (required), branch (default "main"), requirements (bool)

  3. PLUGINS_PIP — extra pip packages your workflows need.

Copy this file to plugins.py and edit to manage your plugins.
"""

PLUGINS_REGISTRY = [
    # ── Required for LTX 2.3 All-in-One v3.0 workflow ──
    "rgthree-comfy",
    "comfyui-custom-scripts",
    "comfyui-impact-pack",
    "comfyui-kjnodes",
    "comfyui-videohelpersuite",
    # ── Add more comfy-cli node IDs here ──
]

PLUGINS_GIT = [
    # Plugins installed via git clone + optional pip install.
    # {
    #     "url": "https://github.com/user/custom-node-repo",
    #     "branch": "main",
    #     "requirements": True,   # run pip install -r requirements.txt
    # },
]

PLUGINS_PIP = [
    # Extra pip packages needed by your workflows.
    # "opencv-python-headless",
    # "numpy<2.0",
]
