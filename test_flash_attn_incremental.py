"""
Test flash-attn installation incrementally with our working base.
"""

import modal
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
modal.enable_output()

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

# Start with our working production base
base_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install([
        "git",
        "ffmpeg",
        "build-essential",
        "ninja-build",  # Important for flash-attn
        "wget",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
    ])
    .pip_install(
        "torch==2.4.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
)

# Method 1: Try installing from source with proper setup
flash_method_1 = (
    base_image
    # Ensure all build dependencies are present
    .pip_install(
        "ninja",
        "packaging",
        "wheel",
        "setuptools",
    )
    # Clone and install from source
    .run_commands(
        # Clone the repository
        "cd /tmp && git clone https://github.com/Dao-AILab/flash-attention.git",
        # Initialize submodules
        "cd /tmp/flash-attention && git submodule update --init --recursive",
        # Install with no-build-isolation
        "cd /tmp/flash-attention && pip install . --no-build-isolation",
        # Cleanup
        "rm -rf /tmp/flash-attention",
    )
)

# Method 2: Try pre-built wheel for our configuration
# For Python 3.10, CUDA 12.1, PyTorch 2.4.1
flash_method_2 = (
    base_image
    .pip_install(
        "packaging",
        "ninja",
    )
    # Try installing with no-build-isolation from PyPI
    .pip_install(
        "flash-attn==2.6.1",
        extra_options="--no-build-isolation",
    )
)

# Choose which method to test
TEST_METHOD = 1  # Change to 2 to test the other method

if TEST_METHOD == 1:
    test_image = flash_method_1
    print("Testing Method 1: Install from source with git clone")
else:
    test_image = flash_method_2
    print("Testing Method 2: Install from PyPI with no-build-isolation")

app = modal.App(f"test-flash-method-{TEST_METHOD}")

@app.function(
    image=test_image,
    gpu="t4",
    timeout=600,  # 10 minutes
)
def test_flash_attn():
    """Test flash-attn installation."""
    import torch
    
    print("="*60)
    print("Flash-Attn Installation Test")
    print("="*60)
    
    # Basic checks
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Test flash-attn
    print("\nTesting flash-attn import...")
    try:
        import flash_attn
        print(f"✅ flash-attn version: {flash_attn.__version__}")
        
        # Test the actual attention function
        from flash_attn import flash_attn_func
        
        # Small test
        batch, seqlen, nheads, headdim = 2, 64, 8, 64
        q = torch.randn(batch, seqlen, nheads, headdim).cuda().half()
        k = torch.randn(batch, seqlen, nheads, headdim).cuda().half()
        v = torch.randn(batch, seqlen, nheads, headdim).cuda().half()
        
        print("\nRunning flash attention...")
        out = flash_attn_func(q, k, v)
        print(f"✅ Flash attention output shape: {out.shape}")
        print(f"✅ Flash attention working correctly!")
        
        return {"success": True, "version": flash_attn.__version__}
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return {"success": False, "error": "import_failed", "message": str(e)}
    except Exception as e:
        print(f"❌ Runtime error: {e}")
        return {"success": False, "error": "runtime_failed", "message": str(e)}

if __name__ == "__main__":
    print(f"\nBuilding and testing flash-attn installation...")
    print("This may take several minutes for compilation...\n")
    
    try:
        with app.run():
            result = test_flash_attn.remote()
            
            print("\n" + "="*60)
            if result["success"]:
                print(f"✅ SUCCESS! Method {TEST_METHOD} works!")
                print(f"flash-attn {result['version']} installed and working")
            else:
                print(f"❌ FAILED! Method {TEST_METHOD} didn't work")
                print(f"Error: {result.get('error')}")
                print(f"Message: {result.get('message')}")
                
    except Exception as e:
        print(f"\n❌ Build/execution failed: {type(e).__name__}")
        print(f"Error: {str(e)[:500]}")
        print("\nThe build might have timed out during compilation.")
        print("Flash-attn compilation typically takes 5-10 minutes.")