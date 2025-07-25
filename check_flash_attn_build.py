"""
Check what happens during flash-attn build with detailed logging.
"""

import modal
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
modal.enable_output()
os.environ["MODAL_LOGLEVEL"] = "DEBUG"

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

# Simple test to see the build process
test_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install([
        "git",
        "build-essential", 
        "ninja-build",
    ])
    .pip_install("torch==2.4.1", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("packaging", "ninja", "wheel")
    # Try the simplest approach first
    .run_commands(
        "pip install flash-attn==2.6.1 --no-build-isolation -v",
    )
)

app = modal.App("check-flash-build")

@app.function(image=test_image)
def check():
    return "Build completed"

if __name__ == "__main__":
    print("Checking flash-attn build process...")
    print("Watch for build logs below:")
    print("-" * 60)
    
    with app.run():
        result = check.remote()
        print(f"\n{result}")