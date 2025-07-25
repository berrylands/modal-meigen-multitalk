"""
Production Modal Image for MeiGen-MultiTalk
Successfully tested with all required ML dependencies
"""

import modal

# Production image with all verified dependencies
multitalk_image = (
    modal.Image.debian_slim(python_version="3.10")
    # System dependencies
    .apt_install([
        "git",
        "ffmpeg",
        "build-essential",
        "libsm6",
        "libxext6", 
        "libxrender-dev",
        "libgomp1",
    ])
    # PyTorch with CUDA 12.1 (matching Colab)
    .pip_install(
        "torch==2.4.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    # Core dependencies
    .pip_install(
        "transformers==4.49.0",  # Pre-CVE version from Colab
        "huggingface_hub",
        "numpy==1.26.4",
        "tqdm",
        "boto3",
    )
    # xformers for attention optimization (instead of flash-attn)
    .pip_install(
        "xformers==0.0.28",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    # Core ML packages
    .pip_install(
        "peft",
        "accelerate",
        "einops",
        "omegaconf",
    )
    # Audio processing
    .pip_install(
        "librosa",
        "soundfile",
        "scipy",
    )
    # Video processing
    .pip_install(
        "opencv-python",
        "moviepy",
        "imageio",
        "imageio-ffmpeg",
    )
    # Diffusion models and utilities
    .pip_install(
        "diffusers>=0.30.0",
        "Pillow",
        "numba==0.59.1",
        "psutil",
        "packaging",
        "ninja",
    )
    # Clone MultiTalk repository
    .run_commands(
        "cd /root && git clone https://github.com/MeiGen-AI/MultiTalk.git",
    )
    # Environment setup
    .env({
        "PYTHONPATH": "/root/MultiTalk",
        "TORCH_CUDA_ARCH_LIST": "7.0;7.5;8.0;8.6;8.9;9.0",
        "CUDA_VISIBLE_DEVICES": "0",
    })
)

# Export for use in app.py
multitalk_image_production = multitalk_image

if __name__ == "__main__":
    print("Production Modal image for MeiGen-MultiTalk")
    print("\nThis image includes:")
    print("✅ PyTorch 2.4.1 with CUDA 12.1")
    print("✅ Transformers 4.49.0 (pre-CVE version)")
    print("✅ xformers 0.0.28 for attention optimization")
    print("✅ All audio/video processing libraries")
    print("✅ Diffusers for image generation")
    print("✅ MultiTalk repository cloned")
    print("\nAll layers have been tested and verified working on GPU.")