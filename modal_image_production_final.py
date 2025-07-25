"""
Final Production Modal Image for MeiGen-MultiTalk
Matches the exact Colab implementation including flash-attn 2.6.1
"""

import modal

# Use CUDA development base for flash-attn compilation
# This matches what Google Colab provides
multitalk_image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime",
        add_python="3.10"
    )
    # System dependencies required for compilation and runtime
    .apt_install([
        # Build tools for flash-attn
        "git",
        "build-essential",
        "ninja-build",
        "cmake",
        "gcc",
        "g++",
        # Runtime dependencies
        "ffmpeg",
        "libsm6",
        "libxext6", 
        "libxrender-dev",
        "libgomp1",
        "wget",
    ])
    # EXACT PyTorch versions from Colab (should already be in base image)
    # Verify and install if needed
    .run_commands(
        "pip list | grep torch || pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121",
    )
    # xformers - EXACT version from Colab
    .pip_install(
        "xformers==0.0.28",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    # Dependencies needed for flash-attn compilation
    .pip_install(
        "packaging",
        "ninja",
        "wheel",
        "setuptools",
    )
    # flash-attn - EXACT version from Colab (2.6.1)
    # Using the same command as in Colab
    .pip_install(
        "flash-attn==2.6.1",
        extra_options="--no-build-isolation",
    )
    # transformers - EXACT version from Colab (pre-CVE)
    .pip_install(
        "transformers==4.49.0",
    )
    # Other critical dependencies
    .pip_install(
        "huggingface_hub",
        "numpy==1.26.4",
        "numba==0.59.1",
        "peft",
        "accelerate",
        "einops",
        "omegaconf",
        "librosa",
        "moviepy",
        "opencv-python",
        "soundfile",
        "scipy",
        "Pillow",
        "diffusers>=0.30.0",
        "tqdm",
        "psutil",
        "boto3",
        "imageio",
        "imageio-ffmpeg",
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

# Alternative: If compilation fails, use this version without flash-attn
# The Colab shows both xformers and flash-attn, so xformers alone might work
multitalk_image_no_flash = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install([
        "git",
        "ffmpeg",
        "build-essential",
        "libsm6",
        "libxext6", 
        "libxrender-dev",
        "libgomp1",
    ])
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "torchaudio==2.4.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "xformers==0.0.28",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers==4.49.0",
        "huggingface_hub",
        "numpy==1.26.4",
        "numba==0.59.1",
        "peft",
        "accelerate",
        "einops",
        "omegaconf",
        "librosa",
        "moviepy",
        "opencv-python",
        "soundfile",
        "scipy",
        "Pillow",
        "diffusers>=0.30.0",
        "tqdm",
        "psutil",
        "boto3",
        "imageio",
        "imageio-ffmpeg",
    )
    .run_commands(
        "cd /root && git clone https://github.com/MeiGen-AI/MultiTalk.git",
    )
    .env({
        "PYTHONPATH": "/root/MultiTalk",
        "TORCH_CUDA_ARCH_LIST": "7.0;7.5;8.0;8.6;8.9;9.0",
        "CUDA_VISIBLE_DEVICES": "0",
    })
)

if __name__ == "__main__":
    print("Final Production Modal Images for MeiGen-MultiTalk")
    print("\nOption 1: multitalk_image (with flash-attn 2.6.1)")
    print("- Uses PyTorch CUDA base image for compilation")
    print("- Includes flash-attn 2.6.1 as in Colab")
    print("- Requires Ampere+ GPU (A100, A10G)")
    print("\nOption 2: multitalk_image_no_flash (xformers only)")
    print("- Uses standard debian base")
    print("- Relies on xformers 0.0.28 for attention")
    print("- Works on all GPUs")
    print("\nBoth match Colab versions exactly where applicable.")