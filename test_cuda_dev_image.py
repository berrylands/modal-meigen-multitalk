"""
Test if CUDA dev image works properly in Modal.
"""

import modal
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
modal.enable_output()

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

# Test basic CUDA dev image
cuda_dev_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04",
        add_python="3.10"
    )
    .pip_install("torch==2.4.1", index_url="https://download.pytorch.org/whl/cu121")
)

app = modal.App("test-cuda-dev")

@app.function(image=cuda_dev_image, gpu="t4")
def test_cuda_dev():
    """Test CUDA development environment."""
    import subprocess
    import torch
    import os
    
    results = {}
    
    # Check CUDA
    results["cuda_available"] = torch.cuda.is_available()
    results["torch_version"] = torch.__version__
    
    # Check nvcc
    try:
        nvcc_result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        results["nvcc_available"] = nvcc_result.returncode == 0
        if results["nvcc_available"]:
            results["nvcc_version"] = nvcc_result.stdout.split('\n')[3]
    except:
        results["nvcc_available"] = False
    
    # Check CUDA_HOME
    results["cuda_home"] = os.environ.get("CUDA_HOME", "Not set")
    
    return results

if __name__ == "__main__":
    print("Testing CUDA development image...")
    
    with app.run():
        result = test_cuda_dev.remote()
        
        print(f"\nCUDA available: {result['cuda_available']}")
        print(f"PyTorch version: {result['torch_version']}")
        print(f"nvcc available: {result['nvcc_available']}")
        if result.get('nvcc_version'):
            print(f"nvcc version: {result['nvcc_version']}")
        print(f"CUDA_HOME: {result['cuda_home']}")