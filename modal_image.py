"""
Modal Image definition for MeiGen-MultiTalk
Based on insights from the Colab notebook
"""

import modal

def download_models():
    """Download all required models during image build."""
    import subprocess
    import os
    
    print("Downloading models...")
    
    # Create weights directory
    os.makedirs("/root/weights", exist_ok=True)
    
    # Download models using huggingface-cli
    models = [
        ("Wan-AI/Wan2.1-I2V-14B-480P", "/root/weights/Wan2.1-I2V-14B-480P"),
        ("TencentGameMate/chinese-wav2vec2-base", "/root/weights/chinese-wav2vec2-base"),
        ("MeiGen-AI/MeiGen-MultiTalk", "/root/weights/MeiGen-MultiTalk"),
    ]
    
    for repo_id, local_dir in models:
        print(f"Downloading {repo_id}...")
        subprocess.run([
            "huggingface-cli", "download", repo_id,
            "--local-dir", local_dir,
            "--local-dir-use-symlinks", "False"
        ], check=True)
    
    # Set up MultiTalk weights (copy into base model directory)
    print("Setting up MultiTalk weights...")
    import shutil
    
    # Backup original index
    base_index = "/root/weights/Wan2.1-I2V-14B-480P/diffusion_pytorch_model.safetensors.index.json"
    if os.path.exists(base_index):
        shutil.move(base_index, base_index + "_old")
    
    # Copy MultiTalk files
    shutil.copy(
        "/root/weights/MeiGen-MultiTalk/diffusion_pytorch_model.safetensors.index.json",
        "/root/weights/Wan2.1-I2V-14B-480P/"
    )
    shutil.copy(
        "/root/weights/MeiGen-MultiTalk/multitalk.safetensors",
        "/root/weights/Wan2.1-I2V-14B-480P/"
    )
    
    print("Model setup complete!")

# Create the Modal image with all dependencies
multitalk_image = (
    modal.Image.debian_slim(python_version="3.10")
    # Install system dependencies first
    .apt_install([
        "git",  # Required for flash-attn installation
        "ffmpeg",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
        "wget",
        "build-essential",  # Required for compiling flash-attn
        "ninja-build",  # Required for flash-attn
    ])
    # Install PyTorch with CUDA 12.1 support (matching Colab)
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1", 
        "torchaudio==2.4.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    # Install xformers with CUDA 12.1
    .pip_install(
        "xformers==0.0.28",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    # Install flash-attn (matching Colab version)
    # First install dependencies needed by flash-attn
    .pip_install(
        "ninja",  # Python package needed for flash-attn build
        "packaging",  # Also needed for flash-attn
    )
    .pip_install(
        "flash-attn==2.6.1",
        extra_options="--no-build-isolation",
    )
    # Install transformers (CRITICAL: pre-CVE version)
    .pip_install(
        "transformers==4.49.0",
        "peft",
        "accelerate",
    )
    # Clone MultiTalk repository
    .run_commands(
        "cd /root && git clone https://github.com/MeiGen-AI/MultiTalk.git",
        "cd /root/MultiTalk && pip install -r requirements.txt",
    )
    # Install additional dependencies
    .pip_install(
        "huggingface_hub",
        "ninja",
        "psutil",
        "packaging",
        "librosa",
        "moviepy",
        "opencv-python",
        "Pillow",
        "diffusers>=0.30.0",
        "numpy==1.26.4",
        "numba==0.59.1",
        "boto3",  # For S3 integration
        "tqdm",
        "scipy",
        "soundfile",
        "misaki[en]",  # G2P engine for TTS (English support)
    )
    # Set environment variables
    .env({
        "PYTHONPATH": "/root/MultiTalk",
        "TORCH_CUDA_ARCH_LIST": "7.0;7.5;8.0;8.6;8.9;9.0",  # Support various GPU architectures
        "CUDA_VISIBLE_DEVICES": "0",
    })
    # Download models during image build (optional - can be done at runtime)
    # Uncomment the next line to include models in the image (warning: large image size)
    # .run_function(download_models, secret=modal.Secret.from_name("huggingface-secret"))
)

# Alternative lighter image without pre-downloaded models
multitalk_image_light = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install([
        "git",  # Required for flash-attn installation
        "ffmpeg",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
        "wget",
        "build-essential",  # Required for compiling flash-attn
        "ninja-build",  # Required for flash-attn
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
    # Skip flash-attn for now - it's causing build issues
    # .pip_install(
    #     "flash-attn==2.6.1",
    #     extra_options="--no-build-isolation",
    # )
    .pip_install(
        "transformers==4.49.0",
        "peft",
        "accelerate",
        "huggingface_hub",
        "ninja",
        "psutil",
        "packaging",
        "librosa",
        "moviepy",
        "opencv-python",
        "Pillow",
        "diffusers>=0.30.0",
        "numpy==1.26.4",
        "numba==0.59.1",
        "boto3",
        "tqdm",
        "scipy",
        "soundfile",
        "misaki[en]",  # G2P engine for TTS (English support)
    )
    .run_commands(
        "cd /root && git clone https://github.com/MeiGen-AI/MultiTalk.git",
        "cd /root/MultiTalk && pip install -r requirements.txt",
    )
    .env({
        "PYTHONPATH": "/root/MultiTalk",
        "TORCH_CUDA_ARCH_LIST": "7.0;7.5;8.0;8.6;8.9;9.0",
        "CUDA_VISIBLE_DEVICES": "0",
    })
)

if __name__ == "__main__":
    print("Modal image definitions created successfully!")
    print("Use 'multitalk_image' for image with pre-downloaded models (large)")
    print("Use 'multitalk_image_light' for lighter image (downloads models at runtime)")