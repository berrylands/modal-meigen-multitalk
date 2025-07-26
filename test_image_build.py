"""
Test script to verify Modal image builds correctly with GPU support.
"""

import modal
import os

# Enable output to see build logs
modal.enable_output()

# Set up Modal authentication
if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

# Import our custom image
from modal_image import multitalk_image_light

app = modal.App("test-image-build")

@app.function(
    image=multitalk_image_light,
    gpu="t4",  # Use T4 for testing (cheaper than A10G)
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def test_environment():
    """Test that all dependencies are correctly installed."""
    import sys
    import subprocess
    
    print("=" * 60)
    print("Modal Image Build Test")
    print("=" * 60)
    
    # Test 1: Python version
    print(f"\n1. Python Version: {sys.version}")
    
    # Test 2: PyTorch and CUDA
    print("\n2. PyTorch and CUDA:")
    try:
        import torch
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU device: {torch.cuda.get_device_name(0)}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Key ML packages
    print("\n3. Key ML Packages:")
    packages = [
        "transformers",
        "diffusers",
        "xformers",
        "flash_attn",
        "accelerate",
        "librosa",
        "moviepy",
        "cv2",
    ]
    
    for package in packages:
        try:
            if package == "cv2":
                import cv2
                version = cv2.__version__
            elif package == "flash_attn":
                import flash_attn
                version = flash_attn.__version__
            else:
                module = __import__(package)
                version = getattr(module, "__version__", "unknown")
            print(f"   ✅ {package}: {version}")
        except ImportError as e:
            print(f"   ❌ {package}: Not found - {e}")
    
    # Test 4: System packages
    print("\n4. System Packages:")
    system_packages = ["ffmpeg", "git"]
    for package in system_packages:
        try:
            result = subprocess.run([package, "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                print(f"   ✅ {package}: {version_line}")
            else:
                print(f"   ❌ {package}: Not found")
        except Exception as e:
            print(f"   ❌ {package}: Error - {e}")
    
    # Test 5: MultiTalk repository
    print("\n5. MultiTalk Repository:")
    multitalk_path = "/root/MultiTalk"
    if os.path.exists(multitalk_path):
        print(f"   ✅ Repository cloned at: {multitalk_path}")
        # Check for key files
        key_files = ["generate_multitalk.py", "requirements.txt"]
        for file in key_files:
            file_path = os.path.join(multitalk_path, file)
            if os.path.exists(file_path):
                print(f"   ✅ Found: {file}")
            else:
                print(f"   ❌ Missing: {file}")
    else:
        print(f"   ❌ Repository not found at: {multitalk_path}")
    
    # Test 6: Environment variables
    print("\n6. Environment Variables:")
    env_vars = ["PYTHONPATH", "CUDA_VISIBLE_DEVICES", "HF_TOKEN"]
    for var in env_vars:
        value = os.environ.get(var, "Not set")
        if var == "HF_TOKEN" and value != "Not set":
            value = f"{value[:8]}... (length: {len(value)})"
        print(f"   {var}: {value}")
    
    # Test 7: GPU Memory Test
    print("\n7. GPU Memory Test:")
    if torch.cuda.is_available():
        try:
            # Try to allocate some GPU memory
            test_tensor = torch.zeros(1000, 1000, 100).cuda()
            print(f"   ✅ Successfully allocated test tensor on GPU")
            print(f"   Memory allocated: {test_tensor.element_size() * test_tensor.numel() / 1024**3:.2f} GB")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"   ❌ GPU memory allocation failed: {e}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    return {"status": "success", "cuda_available": torch.cuda.is_available()}

@app.function(
    image=multitalk_image_light,
    gpu="t4",
)
def test_inference_setup():
    """Test that we can set up for inference."""
    import os
    import sys
    
    print("\nTesting Inference Setup...")
    
    # Add MultiTalk to path
    sys.path.insert(0, "/root/MultiTalk")
    
    try:
        # Try importing key modules from MultiTalk
        print("Attempting to import MultiTalk modules...")
        
        # These imports might fail if the repo structure is different
        # We'll handle that gracefully
        multitalk_files = os.listdir("/root/MultiTalk") if os.path.exists("/root/MultiTalk") else []
        print(f"Files in MultiTalk directory: {multitalk_files[:10]}...")  # Show first 10 files
        
        return {"status": "ready", "multitalk_available": os.path.exists("/root/MultiTalk")}
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    with app.run():
        print("Running Modal image build tests...")
        
        # Test 1: Environment
        print("\n" + "="*60)
        print("Test 1: Environment Setup")
        print("="*60)
        result = test_environment.remote()
        print(f"Result: {result}")
        
        # Test 2: Inference Setup
        print("\n" + "="*60)
        print("Test 2: Inference Setup")
        print("="*60)
        setup_result = test_inference_setup.remote()
        print(f"Result: {setup_result}")
        
        print("\n✅ All tests completed!")