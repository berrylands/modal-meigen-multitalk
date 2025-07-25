"""
Test flash-attn installation using official pre-built wheel.
"""

import modal
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
modal.enable_output()

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

# Use official pre-built wheel for our configuration
flash_attn_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install([
        "git",
        "ffmpeg",
        "build-essential",
    ])
    # Install PyTorch first
    .pip_install(
        "torch==2.4.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    # Install dependencies
    .pip_install(
        "packaging",
        "ninja",
    )
    # Install flash-attn from official pre-built wheel
    .pip_install(
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.2/flash_attn-2.8.2+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    )
)

app = modal.App("test-flash-wheel")

@app.function(
    image=flash_attn_image,
    gpu="a10g",  # Ampere architecture required for flash-attn
)
def test_flash_attn():
    """Test flash-attn from pre-built wheel."""
    import torch
    
    print("="*60)
    print("Flash-Attn Pre-built Wheel Test")
    print("="*60)
    
    # Environment info
    print(f"\nEnvironment:")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Test flash-attn
    print("\nTesting flash-attn...")
    try:
        import flash_attn
        print(f"✅ flash-attn version: {flash_attn.__version__}")
        
        # Test the core functionality
        from flash_attn import flash_attn_func
        
        # Create test tensors
        batch_size = 2
        seq_len = 128
        n_heads = 8
        head_dim = 64
        
        q = torch.randn(batch_size, seq_len, n_heads, head_dim).cuda().half()
        k = torch.randn(batch_size, seq_len, n_heads, head_dim).cuda().half()
        v = torch.randn(batch_size, seq_len, n_heads, head_dim).cuda().half()
        
        print("\nRunning flash attention...")
        out = flash_attn_func(q, k, v)
        
        print(f"✅ Output shape: {out.shape}")
        print(f"✅ Output dtype: {out.dtype}")
        print(f"\n✅ FLASH-ATTN WORKING CORRECTLY!")
        
        # Also test if we can use it with transformers-style input
        print("\nTesting with different tensor format...")
        # BHSD format (batch, heads, seq, dim)
        q2 = torch.randn(batch_size, n_heads, seq_len, head_dim).cuda().half()
        k2 = torch.randn(batch_size, n_heads, seq_len, head_dim).cuda().half()
        v2 = torch.randn(batch_size, n_heads, seq_len, head_dim).cuda().half()
        
        # Transpose to BSHD for flash_attn
        q2_t = q2.transpose(1, 2)
        k2_t = k2.transpose(1, 2)
        v2_t = v2.transpose(1, 2)
        
        out2 = flash_attn_func(q2_t, k2_t, v2_t)
        print(f"✅ Alternative format also works: {out2.shape}")
        
        return {
            "success": True,
            "version": str(flash_attn.__version__),
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
        }
        
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    print("Testing flash-attn with official pre-built wheel...")
    
    with app.run():
        result = test_flash_attn.remote()
        
        print("\n" + "="*60)
        if result["success"]:
            print("✅ SUCCESS!")
            print(f"flash-attn {result['version']} is working correctly")
            print(f"PyTorch {result['torch_version']} with CUDA {result['cuda_version']}")
            print("\nWe can now update the production image with this configuration.")
        else:
            print("❌ FAILED!")
            print(f"Error: {result.get('error')}")