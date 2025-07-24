"""
Check the status of our Modal image builds.
"""

import modal
import os

# Enable output
modal.enable_output()

# Set up Modal authentication
if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

# Try to use the image from modal_image.py
from modal_image import multitalk_image_light

app = modal.App("check-image-status")

@app.function(image=multitalk_image_light, gpu="t4")
def check_status():
    """Simple function to check if image works."""
    import torch
    return {
        "status": "Image loaded successfully!",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available()
    }

if __name__ == "__main__":
    with app.run():
        print("Checking Modal image status...")
        try:
            result = check_status.remote()
            print(f"✅ Success: {result}")
        except Exception as e:
            print(f"❌ Error: {type(e).__name__}: {str(e)}")
            if "build logs" in str(e):
                print("\nThe image build failed. This could be due to:")
                print("1. Package installation errors")
                print("2. Incompatible dependency versions")
                print("3. Network issues downloading packages")
                print("\nTry checking the Modal dashboard for detailed build logs.")