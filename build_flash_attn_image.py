#!/usr/bin/env python3
"""
Build just the image with flash-attn to see the build logs.
"""

import modal
import os

modal.enable_output()

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

# Build image step by step to see where it fails
print("Building Modal image with flash-attn...")

# Start with CUDA base image
image = modal.Image.from_registry(
    "nvidia/cuda:12.1.0-devel-ubuntu22.04",
    add_python="3.10"
)

print("Step 1: Base CUDA image")

# Install system dependencies
image = image.apt_install([
    "git",
    "ffmpeg",
    "libsm6",
    "libxext6",
    "libxrender-dev",
    "libgomp1",
    "wget",
])

print("Step 2: System dependencies")

# Install PyTorch
image = image.pip_install(
    "torch==2.4.1",
    "torchvision==0.19.1", 
    "torchaudio==2.4.1",
    index_url="https://download.pytorch.org/whl/cu121",
)

print("Step 3: PyTorch")

# Install prerequisites
image = image.pip_install(
    "ninja",
    "packaging",
    "wheel",
    "setuptools",
)

print("Step 4: Build prerequisites")

# Try installing flash-attn
image = image.run_commands(
    "echo 'System info:'",
    "python --version",
    "nvcc --version",
    "echo 'Installing flash-attn...'",
    "pip install flash-attn==2.6.1 --no-build-isolation -v || echo 'Flash-attn installation failed'",
)

print("Step 5: Flash-attn installation")

# Create a simple test function
app = modal.App("flash-attn-test")

@app.function(image=image, gpu="a100")
def test_flash_attn():
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        import flash_attn
        print(f"Flash-attn: {flash_attn.__version__}")
        return {"success": True, "flash_attn_version": flash_attn.__version__}
    except Exception as e:
        print(f"Flash-attn import failed: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    with app.run():
        print("\nTesting flash-attn installation...")
        result = test_flash_attn.remote()
        print(f"\nResult: {result}")