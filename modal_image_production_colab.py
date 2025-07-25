"""
Production Modal Image for MeiGen-MultiTalk
Using EXACT versions from the working Colab implementation
"""

import modal

# Production image matching Colab exactly
multitalk_image = (
    modal.Image.debian_slim(python_version="3.10")
    # System dependencies (from Colab)
    .apt_install([
        "git",
        "ffmpeg",
        "build-essential",
        "libsm6",
        "libxext6", 
        "libxrender-dev",
        "libgomp1",
        "wget",
        "ninja-build",
    ])
    # PyTorch - EXACT versions from Colab
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "torchaudio==2.4.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    # xformers - EXACT version from Colab
    .pip_install(
        "xformers==0.0.28",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    # flash-attn - EXACT version from Colab (2.6.1)
    # Using pre-built wheel to avoid compilation timeout
    .pip_install(
        "packaging",
        "ninja",
    )
    .pip_install(
        "flash-attn==2.6.1",
        extra_options="--no-build-isolation",
    )
    # transformers - EXACT version from Colab (pre-CVE)
    .pip_install(
        "transformers==4.49.0",
    )
    # Other dependencies from MultiTalk requirements
    .pip_install(
        "huggingface_hub",
        "numpy==1.26.4",  # From our requirements_ml.txt
        "numba==0.59.1",  # NumPy/Numba compatibility
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

# Export for use in app.py
multitalk_image_production = multitalk_image

if __name__ == "__main__":
    print("Production Modal image for MeiGen-MultiTalk")
    print("\nThis image matches the Colab implementation exactly:")
    print("✅ PyTorch 2.4.1 with CUDA 12.1")
    print("✅ flash-attn 2.6.1 (as used in Colab)")
    print("✅ xformers 0.0.28 (as used in Colab)")
    print("✅ Transformers 4.49.0 (pre-CVE version)")
    print("✅ All other dependencies from MultiTalk")
    print("\nNOTE: Requires Ampere GPU (A100, A10G) for flash-attn")