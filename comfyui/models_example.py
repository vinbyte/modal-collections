"""
Models to download during ComfyUI image build.

`model_dir` accepts two styles:
  1. Relative path (recommended for standard ComfyUI model folders):
       "checkpoints"         -> /root/comfy/ComfyUI/models/checkpoints
       "loras/ltx23"         -> /root/comfy/ComfyUI/models/loras/ltx23
  2. Absolute path (use when the target is outside ComfyUI/models/,
     e.g. a custom node's own model directory):
       "/root/comfy/ComfyUI/custom_nodes/ComfyUI-ReActor/models/insightface"

Common subdirs under ComfyUI/models/:
  checkpoints, diffusion_models, vae, loras, text_encoders,
  clip_vision, controlnet, upscale_models, embeddings,
  latent_upscale_models, unet, audio_encoders

Copy this file to models.py and edit to manage your models.
"""

MODELS_HF = [
    # ── LTX 2.3 fp8 checkpoint (29.1 GB) — recommended for L40S ──
    {
        "repo_id": "Lightricks/LTX-2.3-fp8",
        "filename": "ltx-2.3-22b-dev-fp8.safetensors",
        "model_dir": "checkpoints",
    },
    # ── Text encoders ────────────────────────────────────────────
    {
        "repo_id": "Comfy-Org/ltx-2",
        "filename": "split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
        "model_dir": "text_encoders",
    },
    {
        "repo_id": "Kijai/LTX2.3_comfy",
        "filename": "text_encoders/ltx-2.3_text_projection_bf16.safetensors",
        "model_dir": "text_encoders",
    },
    # ── VAEs ─────────────────────────────────────────────────────
    {
        "repo_id": "Kijai/LTX2.3_comfy",
        "filename": "vae/LTX23_audio_vae_bf16.safetensors",
        "model_dir": "vae",
    },
    {
        "repo_id": "Kijai/LTX2.3_comfy",
        "filename": "vae/LTX23_video_vae_bf16.safetensors",
        "model_dir": "vae",
    },
    {
        "repo_id": "Kijai/LTX2.3_comfy",
        "filename": "vae/taeltx2_3.safetensors",
        "model_dir": "vae",
    },
    # ── Latent upscale models ────────────────────────────────────
    {
        "repo_id": "Lightricks/LTX-2.3",
        "filename": "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors",
        "model_dir": "latent_upscale_models",
    },
    {
        "repo_id": "Lightricks/LTX-2.3",
        "filename": "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
        "model_dir": "latent_upscale_models",
    },
    # ── Workflow-required diffusion model variant ─────────────────
    {
        "repo_id": "Kijai/LTX2.3_comfy",
        "filename": "diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_mxfp8_block32.safetensors",
        "model_dir": "diffusion_models",
    },
]

MODELS_URL = [
    # ── Optional: Realism LoRAs (civitai) ────────────────────────
    # Uncomment to enable the photorealism LoRA combo:
    # {
    #     "url": "https://civitai.red/models/2530917/amateur-hour-ltx-23",
    #     "filename": "AmateurHour_01_rank16.safetensors",
    #     "model_dir": "loras",
    # },
    # {
    #     "url": "https://civitai.red/models/2535622?modelVersionId=2849706",
    #     "filename": "LTX2.3_Soft_Enhance.safetensors",
    #     "model_dir": "loras",
    # },
    # {
    #     "url": "https://civitai.red/models/2200329?modelVersionId=2808759",
    #     "filename": "LTX23-GalaxyAce.safetensors",
    #     "model_dir": "loras",
    # },
    # {
    #     "url": "https://civitai.red/models/2535622?modelVersionId=2849716",
    #     "filename": "LTX2.3_Crisp_Enhance.safetensors",
    #     "model_dir": "loras",
    # },
    # {
    #     "url": "https://civitai.red/models/2538555/ltx-23-luxe-sensual?modelVersionId=2852957",
    #     "filename": "Luxe_Sensual.safetensors",
    #     "model_dir": "loras",
    # },
    # ── Optional: Voice cloning LoRA ──────────────────────────────
    # {
    #     "url": "https://huggingface.co/Comfy-Org/ltx-2.3/resolve/main/split_files/loras/ltx-2.3-id-lora-talkvid-3k.safetensors",
    #     "filename": "ltx-2.3-id-lora-talkvid-3k.safetensors",
    #     "model_dir": "loras",
    # },
    # ── Optional: MelBand RoFormer (voice separation) ─────────────
    # {
    #     "url": "https://huggingface.co/Kijai/MelBandRoFormer_comfy/resolve/main/MelBandRoformer_fp16.safetensors",
    #     "filename": "MelBandRoformer_fp16.safetensors",
    #     "model_dir": "diffusion_models",
    # },
]
