"""
Debug Modal image build issues by building step by step.
"""

import modal
import os
import asyncio

# Enable output
modal.enable_output()

# Set up Modal authentication
if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

async def test_build():
    """Test building the image step by step."""
    print("Starting image build test...")
    
    # Step 1: Basic image
    print("\n1. Testing basic Debian image...")
    basic_image = modal.Image.debian_slim(python_version="3.10")
    
    app = modal.App("test-basic")
    
    @app.function(image=basic_image)
    def test_basic():
        import sys
        return f"Python {sys.version}"
    
    async with app.run():
        result = await test_basic.remote.aio()
        print(f"Basic test result: {result}")
    
    # Step 2: With system packages
    print("\n2. Testing with system packages...")
    sys_image = basic_image.apt_install(["git", "ffmpeg"])
    
    app2 = modal.App("test-sys")
    
    @app2.function(image=sys_image)
    def test_sys():
        import subprocess
        git_version = subprocess.run(["git", "--version"], capture_output=True, text=True)
        ffmpeg_version = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        return {
            "git": git_version.stdout.split('\n')[0] if git_version.returncode == 0 else "Not found",
            "ffmpeg": ffmpeg_version.stdout.split('\n')[0] if ffmpeg_version.returncode == 0 else "Not found"
        }
    
    async with app2.run():
        result = await test_sys.remote.aio()
        print(f"System packages test result: {result}")
    
    print("\nDebug test complete!")

if __name__ == "__main__":
    asyncio.run(test_build())