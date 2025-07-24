"""
Working Modal Image definition for MeiGen-MultiTalk
This version works around the flash-attn build issues
"""

import modal

# Create a working image without flash-attn
# Flash-attn is optional and can be added later if needed
multitalk_image_working = (
    modal.Image.debian_slim(python_version="3.10")
    # System dependencies
    .apt_install([
        "git",
        "ffmpeg",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
        "wget",
        "build-essential",
        "ninja-build",
    ])
    # PyTorch with CUDA 12.1
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1", 
        "torchaudio==2.4.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    # xformers for attention optimization (alternative to flash-attn)
    .pip_install(
        "xformers==0.0.28",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    # Core ML packages
    .pip_install(
        "transformers==4.49.0",  # CRITICAL: pre-CVE version from Colab
        "peft",
        "accelerate",
        "huggingface_hub",
    )
    # Audio/Video processing
    .pip_install(
        "librosa",
        "moviepy",
        "opencv-python",
        "soundfile",
        "scipy",
        "Pillow",
    )
    # Diffusion and other ML packages
    .pip_install(
        "diffusers>=0.30.0",
        "einops",
        "omegaconf",
    )
    # Specific versions for compatibility
    .pip_install(
        "numpy==1.26.4",
        "numba==0.59.1",
    )
    # Utilities
    .pip_install(
        "boto3",  # For S3
        "tqdm",
        "psutil",
        "packaging",
        "ninja",
        "imageio",
        "imageio-ffmpeg",
    )
    # Clone and setup MultiTalk
    .run_commands(
        "cd /root && git clone https://github.com/MeiGen-AI/MultiTalk.git",
        # Don't install requirements.txt as it might have conflicting versions
        # We've already installed everything we need above
    )
    # Environment variables
    .env({
        "PYTHONPATH": "/root/MultiTalk:$PYTHONPATH",
        "TORCH_CUDA_ARCH_LIST": "7.0;7.5;8.0;8.6;8.9;9.0",
        "CUDA_VISIBLE_DEVICES": "0",
    })
)

# Export for use in other files
multitalk_image = multitalk_image_working

if __name__ == "__main__":
    print("Working Modal image definition created!")
    print("This version skips flash-attn to avoid build issues.")
    print("xformers provides similar attention optimization benefits.")