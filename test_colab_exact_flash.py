"""
Test flash-attn 2.6.1 installation (exact Colab version).
"""

import modal
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
modal.enable_output()

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

# Test with exact Colab configuration
colab_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install([
        "git",
        "build-essential",
        "ninja-build",
    ])
    # Exact PyTorch from Colab
    .pip_install(
        "torch==2.4.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    # Try pre-built wheel for flash-attn 2.6.1
    .pip_install(
        "packaging",
        "ninja",
    )
    # Check if there's a pre-built wheel for 2.6.1
    .run_commands(
        # First, let's see what's available
        "pip index versions flash-attn",
        # Try to find a pre-built wheel
        "pip install flash-attn==2.6.1 --no-build-isolation --prefer-binary -v",
    )
)

app = modal.App("test-colab-flash")

@app.function(
    image=colab_image,
    gpu="a100",  # Use A100 as specified in Colab
)
def test_flash():
    """Test flash-attn 2.6.1."""
    import torch
    
    print("Testing Colab-exact flash-attn configuration...")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        import flash_attn
        print(f"\n✅ flash-attn {flash_attn.__version__} imported successfully!")
        
        # Quick test
        from flash_attn import flash_attn_func
        b, s, h, d = 2, 64, 8, 64
        q = torch.randn(b, s, h, d).cuda().half()
        k = torch.randn(b, s, h, d).cuda().half()
        v = torch.randn(b, s, h, d).cuda().half()
        out = flash_attn_func(q, k, v)
        print(f"✅ Flash attention working! Output: {out.shape}")
        
        return {"success": True, "version": str(flash_attn.__version__)}
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    print("Testing exact Colab configuration...")
    
    with app.run():
        result = test_flash.remote()
        
        if result["success"]:
            print(f"\n✅ SUCCESS! flash-attn {result['version']} working on A100")
        else:
            print(f"\n❌ Failed: {result['error']}")