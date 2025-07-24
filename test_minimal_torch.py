"""
Test minimal PyTorch image build on Modal.
"""

import modal
import os

# Enable output
modal.enable_output()

# Set up Modal authentication
if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

# Create minimal torch image
minimal_torch_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.4.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
)

app = modal.App("test-minimal-torch")

@app.function(
    image=minimal_torch_image,
    gpu="t4",  # Use cheapest GPU for testing
    timeout=60,
)
def test_torch_gpu():
    """Test PyTorch with GPU."""
    import torch
    
    print("=" * 60)
    print("PyTorch GPU Test")
    print("=" * 60)
    
    device_info = {
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": str(torch.version.cuda) if torch.cuda.is_available() else None,
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        device_info["device_name"] = str(torch.cuda.get_device_name(0))
        device_info["device_memory_gb"] = float(torch.cuda.get_device_properties(0).total_memory / (1024**3))
        
        # Test tensor operations
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            result = torch.sum(test_tensor).item()
            device_info["tensor_test"] = "Success"
        except Exception as e:
            device_info["tensor_test"] = f"Failed: {e}"
    
    for key, value in device_info.items():
        print(f"{key}: {value}")
    
    print("=" * 60)
    return device_info

if __name__ == "__main__":
    print("Building and testing minimal PyTorch image...")
    print("This may take a few minutes on first run...")
    
    with app.run():
        result = test_torch_gpu.remote()
        print("\nTest completed!")
        print(f"Result: {result}")
        
        if result["cuda_available"]:
            print("\n✅ PyTorch with CUDA is working correctly!")
        else:
            print("\n❌ CUDA is not available")