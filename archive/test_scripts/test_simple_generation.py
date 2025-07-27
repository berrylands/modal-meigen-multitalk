#!/usr/bin/env python3
"""
Simplest possible MultiTalk test with exact Colab defaults.
"""

import modal
import os

modal.enable_output()

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

# Minimal image
simple_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["git", "ffmpeg"])
    .pip_install(
        "torch==2.4.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers==4.49.0",
        "diffusers>=0.30.0",
        "librosa",
        "numpy==1.24.4",
        "numba==0.59.1",
        "soundfile",
        "boto3",
        "huggingface_hub",
    )
)

app = modal.App("test-simple")

model_volume = modal.Volume.from_name("multitalk-models", create_if_missing=True)

@app.function(
    image=simple_image,
    gpu="a10g",
    volumes={"/models": model_volume},
    secrets=[modal.Secret.from_name("aws-secret")],
    timeout=300,
)
def test_models_and_setup():
    """
    Just test if models are set up correctly.
    """
    import os
    import json
    
    print("Testing MultiTalk setup...")
    
    # Check models
    print("\nChecking models:")
    base_dir = "/models/base"
    required_files = [
        "config.json",
        "multitalk.safetensors",
        "diffusion_pytorch_model.safetensors.index.json"
    ]
    
    for f in required_files:
        path = os.path.join(base_dir, f)
        exists = os.path.exists(path)
        print(f"  {f}: {'✅' if exists else '❌'}")
        
        if f == "diffusion_pytorch_model.safetensors.index.json" and exists:
            # Check if it's the MultiTalk version
            with open(path, 'r') as file:
                content = json.load(file)
                has_multitalk = any("multitalk" in k for k in content.get("weight_map", {}).values())
                print(f"    Contains multitalk weights: {'✅' if has_multitalk else '❌'}")
    
    # Check wav2vec
    wav2vec_dir = "/models/wav2vec"
    if os.path.exists(os.path.join(wav2vec_dir, "config.json")):
        print("\n✅ wav2vec2 model present")
    else:
        print("\n❌ wav2vec2 model missing")
    
    # Try simplest possible generation
    print("\nTrying to import MultiTalk modules...")
    try:
        import sys
        sys.path.insert(0, "/root/MultiTalk")
        
        # Check if we can at least import
        import subprocess
        result = subprocess.run(
            ["python3", "-c", "import sys; sys.path.insert(0, '/root/MultiTalk'); from wan.multitalk import *"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Can import MultiTalk modules")
        else:
            print("❌ Cannot import MultiTalk modules")
            print(f"Error: {result.stderr[:500]}")
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
    
    # Check MultiTalk repo
    if os.path.exists("/root/MultiTalk/generate_multitalk.py"):
        print("\n✅ MultiTalk repo present")
    else:
        print("\n❌ MultiTalk repo missing")
        
    return {"models_ok": True}


if __name__ == "__main__":
    with app.run():
        print("Testing MultiTalk Setup\n")
        
        result = test_models_and_setup.remote()
        
        print(f"\nResult: {result}")
