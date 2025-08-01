#!/usr/bin/env python3
"""
Debug script to test MultiTalk setup and identify issues.
"""

import modal
import os

app = modal.App("multitalk-debug")

# Use the same image and setup
from app_multitalk_cuda import multitalk_cuda_image, model_volume, hf_cache_volume

@app.function(
    image=multitalk_cuda_image,
    gpu="a10g",
    memory=32768,
    timeout=300,
    secrets=[modal.Secret.from_name("aws-secret")],
    volumes={
        "/models": model_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
)
def debug_setup():
    """Debug the MultiTalk setup and model availability."""
    import torch
    import subprocess
    import sys
    
    print("=== DEBUG: MultiTalk Setup ===")
    
    # Check GPU
    print(f"\n1. GPU Check:")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Check models
    print(f"\n2. Model Check:")
    weights_dir = "/root/MultiTalk/weights"
    if os.path.exists(weights_dir):
        print(f"   Weights directory exists")
        for model_dir in ["Wan2.1-I2V-14B-480P", "chinese-wav2vec2-base", "MeiGen-MultiTalk"]:
            path = os.path.join(weights_dir, model_dir)
            if os.path.exists(path):
                print(f"   ✓ {model_dir} exists")
                # Check size
                size = subprocess.check_output(['du', '-sh', path]).decode().split()[0]
                print(f"     Size: {size}")
            else:
                print(f"   ✗ {model_dir} MISSING")
    else:
        print(f"   ✗ Weights directory does not exist!")
    
    # Check MultiTalk installation
    print(f"\n3. MultiTalk Installation:")
    multitalk_path = "/root/MultiTalk/generate_multitalk.py"
    if os.path.exists(multitalk_path):
        print(f"   ✓ generate_multitalk.py exists")
    else:
        print(f"   ✗ generate_multitalk.py MISSING")
    
    # Check flash-attn
    print(f"\n4. Flash Attention:")
    try:
        import flash_attn
        print(f"   ✓ Version: {flash_attn.__version__}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Try a minimal generation command
    print(f"\n5. Testing minimal generation...")
    os.chdir("/root/MultiTalk")
    
    # Create minimal test files
    with open("test.json", "w") as f:
        f.write('{"prompt": "test", "cond_image": "test.png", "cond_audio": {"person1": "test.wav"}}')
    
    # Just check if the script starts
    result = subprocess.run(
        ["python3", "generate_multitalk.py", "--help"],
        capture_output=True,
        text=True,
        timeout=10
    )
    print(f"   Help command exit code: {result.returncode}")
    if result.returncode != 0:
        print(f"   STDERR: {result.stderr[:500]}")
    
    return {"debug_complete": True}

if __name__ == "__main__":
    with app.run():
        result = debug_setup.remote()
        print(f"Debug result: {result}")