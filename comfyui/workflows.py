"""
Workflow definitions for the OpenAI-compatible video generation API.

Each entry maps a model identifier to a ComfyUI workflow JSON file and
describes which nodes to override with user-provided parameters.

How it works:
  1. The API receives a POST /v1/video/generations request
  2. It loads the workflow JSON template for the specified model
  3. It injects user parameters into the specified node inputs/widgets
  4. It submits the modified workflow to ComfyUI and polls for completion

param_map entries specify how user parameters map to workflow nodes:
  - Key: the API parameter name (e.g. "prompt")
  - Value: the target in the workflow (format varies by node type):
      "NODE_ID.widgets.INDEX"  → widget value (for widgets_values list)
      "NODE_ID.inputs.NAME"    → node input (for node inputs dict)

Example:
  {
      "file": "workflows/my_wf.json",
      "param_map": {
          "prompt": "10.inputs.text",       # node 10, input named "text"
          "width": "15.widgets.0",          # node 15, first widget
          "height": "15.widgets.1",         # node 15, second widget
      },
  }
"""

WORKFLOWS = {
    "ltx2.3": {
        "file": "workflows/ltx23_v30.json",
        "param_map": {
            # CLIPTextEncode node 1879 — "Positive Prompt"
            "prompt": "1879.widgets.0",
        },
    },
}
