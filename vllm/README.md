# vLLM Inference on Modal

OpenAI-compatible LLM inference server using [vLLM](https://docs.vllm.ai/), deployed on [Modal](https://modal.com).

## Current Model

[cyankiwi/Qwen3.6-27B-AWQ-INT4](https://huggingface.co/cyankiwi/Qwen3.6-27B-AWQ-INT4) — AWQ 4-bit quantized Qwen3.6-27B (27B params, ~21 GB VRAM) with vision + reasoning support, running on an A100 80GB GPU at full 262K context length.

## Quick Start

```bash
# Dev mode (ephemeral, logs in terminal)
modal serve vllm/serve.py

# Production (persistent, auto-scaling)
modal deploy vllm/serve.py

# Test the server
modal run vllm/serve.py
modal run vllm/serve.py --content "What is the meaning of life?"
```

## Using the API

Once deployed, Modal prints a URL like `https://your-workspace--vllm-inference-serve.modal.run`. Use it as an OpenAI-compatible endpoint:

### Python (openai SDK)

```bash
pip install openai
```

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-workspace--vllm-inference-serve.modal.run/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="cyankiwi/Qwen3.6-27B-AWQ-INT4",
    max_tokens=4096,
)
print(response.choices[0].message.content)
```

### Streaming with thinking

```python
response = client.chat.completions.create(
    model="cyankiwi/Qwen3.6-27B-AWQ-INT4",
    stream=True,
    max_tokens=8192,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)
for chunk in response:
    delta = chunk.choices[0].delta
    print(delta.content or delta.reasoning_content or "", end="", flush=True)
```

### curl

```bash
curl -X POST "https://your-workspace--vllm-inference-serve.modal.run/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cyankiwi/Qwen3.6-27B-AWQ-INT4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 512
  }'
```

## Configuration

All configurable settings are at the top of `serve.py`:

| Variable | Default | Description |
|---|---|---|
| `GPU_TYPE` | `A100-80GB` | GPU type (A100-80GB, H100, H200, etc.) |
| `N_GPU` | `1` | Number of GPUs per container |
| `MODEL_NAME` | `cyankiwi/Qwen3.6-27B-AWQ-INT4` | HuggingFace model ID |
| `MODEL_REVISION` | `None` | Pin a specific commit hash |
| `MAX_MODEL_LEN` | `262144` | Max context length in tokens |
| `FAST_BOOT` | `False` | Skip CUDA graphs for faster cold start |
| `LANGUAGE_MODEL_ONLY` | `False` | Skip vision encoder for text-only (saves ~1.2 GB) |
| `REASONING_PARSER` | `qwen3` | Reasoning parser for thinking mode |
| `ENABLE_AUTO_TOOL_CHOICE` | `True` | Enable tool use support |
| `TOOL_CALL_PARSER` | `qwen3_coder` | Tool call parser name |
| `SCALEDOWN_WINDOW` | `900` | Seconds to stay up with no requests |
| `MAX_CONCURRENT_INPUTS` | `100` | Max concurrent requests per replica |
| `VLLM_VERSION` | `0.19.0` | vLLM package version |
| `TRANSFORMERS_VERSION` | `5.5.0` | transformers package version |

### GPU Selection Guide

| GPU | VRAM | Best For |
|---|---|---|
| A100-80GB | 80 GB | AWQ INT4 27B at full 262K context, best value |
| H100 | 80 GB | Faster inference than A100, auto-upgraded to H200 |
| H200 | 141 GB | BF16 27B at 262K, or 70B+ models |
| L40S | 48 GB | AWQ INT4 27B at up to ~64K context, cheapest |

### Switching Models

1. Change `MODEL_NAME` in `serve.py`
2. Adjust `MAX_MODEL_LEN` based on the model's context window and GPU VRAM
3. Set `REASONING_PARSER` if the model supports thinking (e.g. `qwen3`, `deepseek_r1`)
4. Update `TRANSFORMERS_VERSION` if the model requires a newer version
5. Redeploy: `modal deploy vllm/serve.py`

## Persistent Storage

| Volume | Mount Path | Purpose |
|---|---|---|
| `huggingface-cache` | `/root/.cache/huggingface` | Model weights from HuggingFace Hub |
| `vllm-cache` | `/root/.cache/vllm` | vLLM JIT compilation artifacts |

These volumes persist across container restarts, so model weights only need to be downloaded once (~19GB for AWQ INT4). Subsequent cold starts are much faster.

## API Endpoints

Once the server is running, it exposes:

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | Chat completion (OpenAI-compatible) |
| `/v1/completions` | POST | Text completion |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |
| `/docs` | GET | Swagger UI (interactive API docs) |

## Troubleshooting

**OOM (Out of Memory)**: Reduce `MAX_MODEL_LEN` (e.g. from 262144 to 131072 or 65536) or switch to a quantized model (AWQ INT4).

**Slow cold start**: Set `FAST_BOOT = True` to skip CUDA graph capture. This trades inference throughput for faster startup. Alternatively, use `min_containers=1` to keep a replica warm.

**Model not found**: Ensure the model ID is correct on [HuggingFace](https://huggingface.co/models). Some models require `TRANSFORMERS_VERSION` to be updated.
