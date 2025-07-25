"""
Test proper flash-attn installation in Modal.
Using the correct approach with CUDA dev image and no-build-isolation.
"""

import modal
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
modal.enable_output()
os.environ["MODAL_LOGLEVEL"] = "DEBUG"

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

# Solution 1: Using NVIDIA CUDA development image
flash_attn_image_v1 = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04",
        add_python="3.10"
    )
    .apt_install([
        "git",
        "ffmpeg",
        "build-essential",
        "ninja-build",
    ])
    # Install PyTorch first
    .pip_install(
        "torch==2.4.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    # Install build dependencies
    .pip_install(
        "ninja",
        "packaging",
        "wheel",
        "setuptools",
    )
    # Install flash-attn with no-build-isolation
    .pip_install(
        "flash-attn==2.6.1",
        extra_options="--no-build-isolation",
    )
)

# Solution 2: Using PyTorch development image
flash_attn_image_v2 = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel",
    )
    .apt_install([
        "git",
        "ffmpeg",
        "build-essential",
        "ninja-build",
    ])
    .pip_install(
        "ninja",
        "packaging",
        "wheel",
    )
    .pip_install(
        "flash-attn==2.6.1",
        extra_options="--no-build-isolation",
    )
)

# Test with Solution 1 (change to 2 to test the other)
TEST_VERSION = 1

if TEST_VERSION == 1:
    test_image = flash_attn_image_v1
    print("Testing Solution 1: NVIDIA CUDA dev image")
else:
    test_image = flash_attn_image_v2
    print("Testing Solution 2: PyTorch dev image")

app = modal.App("test-flash-attn-proper")

@app.function(
    image=test_image,
    gpu="t4",
)
def test_flash_attn():
    """Test flash-attn installation and functionality."""
    import torch
    import sys
    
    results = {
        "python": sys.version.split()[0],
        "cuda_available": False,
        "flash_attn_status": "not_tested",
    }
    
    # Test CUDA
    try:
        results["cuda_available"] = torch.cuda.is_available()
        results["torch_version"] = torch.__version__
        if results["cuda_available"]:
            results["cuda_version"] = torch.version.cuda
            results["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as e:
        results["cuda_error"] = str(e)
    
    # Test flash-attn import
    try:
        import flash_attn
        results["flash_attn_version"] = flash_attn.__version__
        results["flash_attn_status"] = "imported"
        
        # Test flash attention function
        from flash_attn import flash_attn_func
        
        # Create test tensors
        batch_size = 2
        seqlen = 128
        nheads = 8
        headdim = 64
        
        q = torch.randn(batch_size, seqlen, nheads, headdim).cuda().half()
        k = torch.randn(batch_size, seqlen, nheads, headdim).cuda().half()
        v = torch.randn(batch_size, seqlen, nheads, headdim).cuda().half()
        
        # Run flash attention
        out = flash_attn_func(q, k, v)
        results["flash_attn_compute"] = "success"
        results["output_shape"] = str(out.shape)
        
    except ImportError as e:
        results["flash_attn_status"] = "import_failed"
        results["flash_attn_error"] = str(e)
    except Exception as e:
        results["flash_attn_status"] = "compute_failed"
        results["flash_attn_error"] = str(e)[:200]
    
    return results

if __name__ == "__main__":
    print("\nTesting flash-attn proper installation...")
    print("="*60)
    
    try:
        with app.run():
            result = test_flash_attn.remote()
            
            print(f"\nPython: {result['python']}")
            print(f"PyTorch: {result.get('torch_version', 'error')}")
            print(f"CUDA: {result.get('cuda_available', 'error')}")
            if result.get('gpu_name'):
                print(f"GPU: {result['gpu_name']}")
            
            print(f"\nflash-attn status: {result['flash_attn_status']}")
            if result.get('flash_attn_version'):
                print(f"flash-attn version: {result['flash_attn_version']}")
            if result.get('flash_attn_compute') == 'success':
                print(f"✅ Flash attention compute successful!")
                print(f"Output shape: {result['output_shape']}")
            elif result.get('flash_attn_error'):
                print(f"❌ Error: {result['flash_attn_error']}")
            
            success = (
                result.get('cuda_available') and 
                result.get('flash_attn_compute') == 'success'
            )
            
            print("\n" + "="*60)
            if success:
                print("✅ FLASH-ATTN INSTALLATION SUCCESSFUL!")
                print(f"\nSolution {TEST_VERSION} works correctly.")
            else:
                print("❌ FLASH-ATTN INSTALLATION FAILED!")
                print(f"\nSolution {TEST_VERSION} needs adjustment.")
                
    except Exception as e:
        print(f"\n❌ Build failed: {str(e)[:500]}")
        print("\nThis might be a build timeout or compilation issue.")
        print("Flash-attn compilation can take 5-10 minutes.")