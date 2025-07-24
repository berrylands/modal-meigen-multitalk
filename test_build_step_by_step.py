"""
Test Modal image build step by step to find the failing package.
"""

import modal
import os

# Enable output to see build logs
modal.enable_output()

# Set up Modal authentication
if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

def test_package_group(name, base_image, packages):
    """Test a group of packages."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    try:
        if isinstance(packages, dict):
            # pip install with options
            test_image = base_image.pip_install(**packages)
        else:
            # simple pip install
            test_image = base_image.pip_install(*packages)
        
        app = modal.App(f"test-{name.lower().replace(' ', '-')}")
        
        @app.function(image=test_image)
        def test():
            return f"{name} installed successfully"
        
        with app.run():
            result = test.remote()
            print(f"✅ {result}")
            return True
            
    except Exception as e:
        print(f"❌ Failed: {str(e)[:200]}...")
        return False

if __name__ == "__main__":
    print("Testing Modal image build step by step...")
    
    # Base image
    base = modal.Image.debian_slim(python_version="3.10")
    
    # Test 1: System packages
    if test_package_group("System packages", base, 
        base.apt_install(["git", "ffmpeg", "build-essential"])):
        base = base.apt_install(["git", "ffmpeg", "build-essential"])
    
    # Test 2: PyTorch
    if test_package_group("PyTorch", base, {
        "packages": ["torch==2.4.1", "torchvision==0.19.1", "torchaudio==2.4.1"],
        "index_url": "https://download.pytorch.org/whl/cu121"
    }):
        base = base.pip_install(
            "torch==2.4.1", "torchvision==0.19.1", "torchaudio==2.4.1",
            index_url="https://download.pytorch.org/whl/cu121"
        )
    
    # Test 3: xformers
    if test_package_group("xformers", base, {
        "packages": ["xformers==0.0.28"],
        "index_url": "https://download.pytorch.org/whl/cu121"
    }):
        base = base.pip_install("xformers==0.0.28", index_url="https://download.pytorch.org/whl/cu121")
    
    # Test 4: ML core packages
    ml_core = ["transformers==4.49.0", "peft", "accelerate", "huggingface_hub"]
    if test_package_group("ML core packages", base, ml_core):
        base = base.pip_install(*ml_core)
    
    # Test 5: Audio/Video packages
    av_packages = ["librosa", "moviepy", "opencv-python", "soundfile", "scipy"]
    if test_package_group("Audio/Video packages", base, av_packages):
        base = base.pip_install(*av_packages)
    
    # Test 6: Other packages
    other_packages = ["Pillow", "diffusers>=0.30.0", "tqdm", "psutil", "ninja", "packaging"]
    if test_package_group("Other packages", base, other_packages):
        base = base.pip_install(*other_packages)
    
    # Test 7: NumPy/Numba (potential issue)
    numpy_packages = ["numpy==1.26.4", "numba==0.59.1"]
    if test_package_group("NumPy/Numba", base, numpy_packages):
        base = base.pip_install(*numpy_packages)
    
    # Test 8: Additional packages
    additional = ["boto3", "einops", "omegaconf", "imageio", "imageio-ffmpeg"]
    test_package_group("Additional packages", base, additional)
    
    print("\n" + "="*60)
    print("Step-by-step build test completed!")
    print("="*60)