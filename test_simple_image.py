"""
Test a simplified Modal image build to debug issues.
"""

import modal
import os

# Enable output
modal.enable_output()

# Set up Modal authentication
if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

# Create a simplified image for testing
test_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["git", "ffmpeg"])
    .pip_install("torch==2.4.1", index_url="https://download.pytorch.org/whl/cu121")
)

app = modal.App("test-simple-image")

@app.function(image=test_image)
def test_torch():
    """Test that PyTorch is installed."""
    import torch
    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available()
    }

if __name__ == "__main__":
    with app.run():
        print("Testing simplified image...")
        result = test_torch.remote()
        print(f"Result: {result}")