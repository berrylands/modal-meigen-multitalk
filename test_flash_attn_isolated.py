"""
Test flash-attn installation in isolation.
"""

import modal
import os
import sys

# Forcefully enable output
sys.stdout.reconfigure(line_buffering=True)
modal.enable_output()

# Set up Modal authentication
if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

# Test flash-attn installation with proper dependencies
flash_test_image = (
    modal.Image.debian_slim(python_version="3.10")
    # System deps MUST come first
    .apt_install([
        "git",
        "build-essential",
        "ninja-build",
        "cmake",
        "gcc",
        "g++",
    ])
    # PyTorch MUST be installed before flash-attn
    .pip_install(
        "torch==2.4.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    # Dependencies for flash-attn
    .pip_install(
        "ninja",
        "packaging",
        "wheel",
        "setuptools",
    )
    # Now try flash-attn
    .pip_install(
        "flash-attn==2.6.1",
        extra_options="--no-build-isolation",
    )
)

app = modal.App("test-flash-attn-isolated")

@app.function(image=flash_test_image, gpu="t4")
def test_flash():
    """Test if flash-attn imports correctly."""
    results = {}
    
    # Test torch
    try:
        import torch
        results["torch"] = f"✅ {torch.__version__}"
        results["cuda"] = f"✅ CUDA {torch.version.cuda}" if torch.cuda.is_available() else "❌ No CUDA"
    except Exception as e:
        results["torch"] = f"❌ {str(e)}"
    
    # Test flash_attn
    try:
        import flash_attn
        results["flash_attn"] = f"✅ {flash_attn.__version__}"
    except Exception as e:
        results["flash_attn"] = f"❌ {str(e)}"
    
    return results

if __name__ == "__main__":
    print("Testing flash-attn installation...")
    print("Build logs should appear below:")
    print("-" * 60)
    
    try:
        with app.run():
            result = test_flash.remote()
            print("\nTest Results:")
            for key, value in result.items():
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"\n❌ Build failed: {type(e).__name__}")
        print(f"   {str(e)}")
        print("\nThis likely means the image build failed.")
        print("Check the Modal dashboard for detailed build logs.")