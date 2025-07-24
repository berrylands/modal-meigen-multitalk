"""
Progressively test Modal image builds to identify issues.
"""

import modal
import os

# Enable output
modal.enable_output()

# Set up Modal authentication
if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

# Test 1: PyTorch + xformers
torch_xformers_image = (
    modal.Image.debian_slim(python_version="3.10")
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
)

app = modal.App("test-progressive-build")

@app.function(
    image=torch_xformers_image,
    gpu="t4",
    timeout=60,
)
def test_torch_xformers():
    """Test PyTorch + xformers."""
    import torch
    
    results = {
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    
    try:
        import xformers
        results["xformers_version"] = str(xformers.__version__)
        results["xformers_status"] = "Success"
    except Exception as e:
        results["xformers_status"] = f"Failed: {str(e)}"
    
    return results

# Test 2: Add transformers
transformers_image = torch_xformers_image.pip_install(
    "transformers==4.49.0",
    "peft",
    "accelerate",
)

@app.function(
    image=transformers_image,
    gpu="t4",
    timeout=60,
)
def test_transformers():
    """Test with transformers added."""
    import torch
    
    results = {
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    
    try:
        import transformers
        results["transformers_version"] = str(transformers.__version__)
        results["transformers_status"] = "Success"
    except Exception as e:
        results["transformers_status"] = f"Failed: {str(e)}"
    
    return results

# Test 3: Add flash-attn (likely problematic)
flash_attn_image = transformers_image.pip_install(
    "flash-attn==2.6.1",
    extra_options="--no-build-isolation",
)

@app.function(
    image=flash_attn_image,
    gpu="t4",
    timeout=60,
)
def test_flash_attn():
    """Test with flash-attn added."""
    import torch
    
    results = {
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    
    try:
        import flash_attn
        results["flash_attn_version"] = str(flash_attn.__version__)
        results["flash_attn_status"] = "Success"
    except Exception as e:
        results["flash_attn_status"] = f"Failed: {str(e)}"
    
    return results

if __name__ == "__main__":
    print("Testing progressive image builds...")
    
    with app.run():
        # Test 1
        print("\n" + "="*60)
        print("Test 1: PyTorch + xformers")
        print("="*60)
        try:
            result1 = test_torch_xformers.remote()
            print(f"Result: {result1}")
            if result1.get("xformers_status") == "Success":
                print("✅ xformers installed successfully")
            else:
                print("❌ xformers failed")
        except Exception as e:
            print(f"❌ Build failed: {e}")
        
        # Test 2
        print("\n" + "="*60)
        print("Test 2: + transformers")
        print("="*60)
        try:
            result2 = test_transformers.remote()
            print(f"Result: {result2}")
            if result2.get("transformers_status") == "Success":
                print("✅ transformers installed successfully")
            else:
                print("❌ transformers failed")
        except Exception as e:
            print(f"❌ Build failed: {e}")
        
        # Test 3
        print("\n" + "="*60)
        print("Test 3: + flash-attn")
        print("="*60)
        try:
            result3 = test_flash_attn.remote()
            print(f"Result: {result3}")
            if result3.get("flash_attn_status") == "Success":
                print("✅ flash-attn installed successfully")
            else:
                print("❌ flash-attn failed")
        except Exception as e:
            print(f"❌ Build failed: {e}")
        
        print("\n" + "="*60)
        print("Progressive build test completed!")
        print("="*60)