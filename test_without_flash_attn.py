"""
Test Modal image without flash-attn to see if that's the issue.
"""

import modal
import os

# Enable output
modal.enable_output()

# Set up Modal authentication
if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

# Create image without flash-attn
multitalk_image_no_flash = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install([
        "git",
        "ffmpeg",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
        "wget",
        "build-essential",
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
    # Skip flash-attn for now
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
        "einops",
        "omegaconf",
        "imageio",
        "imageio-ffmpeg",
    )
    .run_commands(
        "cd /root && git clone https://github.com/MeiGen-AI/MultiTalk.git",
    )
    .env({
        "PYTHONPATH": "/root/MultiTalk:$PYTHONPATH",
        "TORCH_CUDA_ARCH_LIST": "7.0;7.5;8.0;8.6;8.9;9.0",
        "CUDA_VISIBLE_DEVICES": "0",
    })
)

app = modal.App("test-no-flash-attn")

@app.function(
    image=multitalk_image_no_flash,
    gpu="t4",
    timeout=120,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def test_full_environment():
    """Test the complete environment without flash-attn."""
    import sys
    import os
    import subprocess
    
    print("=" * 60)
    print("Full Environment Test (without flash-attn)")
    print("=" * 60)
    
    results = {}
    
    # Python version
    results["python_version"] = sys.version.split()[0]
    
    # PyTorch
    try:
        import torch
        results["torch_version"] = str(torch.__version__)
        results["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            results["gpu_name"] = str(torch.cuda.get_device_name(0))
    except Exception as e:
        results["torch_error"] = str(e)
    
    # Key packages
    packages = [
        "transformers", "diffusers", "xformers", "accelerate",
        "librosa", "moviepy", "cv2", "einops", "omegaconf"
    ]
    
    for package in packages:
        try:
            if package == "cv2":
                import cv2
                results[f"{package}_version"] = cv2.__version__
            else:
                module = __import__(package)
                results[f"{package}_version"] = getattr(module, "__version__", "installed")
        except ImportError:
            results[f"{package}_version"] = "not found"
    
    # MultiTalk repo
    results["multitalk_exists"] = os.path.exists("/root/MultiTalk")
    
    # HF token
    results["hf_token_available"] = any(key in os.environ for key in ["HF_TOKEN", "HUGGINGFACE_TOKEN"])
    
    # Print results
    for key, value in results.items():
        print(f"{key}: {value}")
    
    print("=" * 60)
    
    # Check if we need flash-attn
    print("\nChecking MultiTalk requirements...")
    try:
        with open("/root/MultiTalk/requirements.txt", "r") as f:
            requirements = f.read()
            if "flash" in requirements.lower():
                print("⚠️  WARNING: MultiTalk requirements mention flash-attn")
                print("This may cause issues during inference")
            else:
                print("✅ No flash-attn requirement found in MultiTalk")
    except:
        print("Could not read MultiTalk requirements")
    
    return results

if __name__ == "__main__":
    print("Testing full environment without flash-attn...")
    print("This may take several minutes on first run...")
    
    with app.run():
        try:
            result = test_full_environment.remote()
            print("\n✅ Environment test completed successfully!")
            
            # Check critical components
            if result.get("cuda_available"):
                print(f"✅ GPU available: {result.get('gpu_name')}")
            else:
                print("❌ GPU not available")
            
            if result.get("multitalk_exists"):
                print("✅ MultiTalk repository cloned")
            else:
                print("❌ MultiTalk repository missing")
            
            if result.get("hf_token_available"):
                print("✅ HuggingFace token available")
            else:
                print("❌ HuggingFace token missing")
                
        except Exception as e:
            print(f"\n❌ Test failed: {e}")