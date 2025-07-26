"""
Test Modal image layers incrementally.
Each test is a separate script to avoid decorator scope issues.
"""

import modal
import os
import sys

# Enable debugging
sys.stdout.reconfigure(line_buffering=True)
modal.enable_output()

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

# Base image that we know works
BASE_IMAGE = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install([
        "git",
        "ffmpeg", 
        "build-essential",
    ])
    .pip_install(
        "torch==2.4.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install("transformers==4.49.0")
    .pip_install("huggingface_hub")
    .pip_install("numpy==1.26.4")
    .pip_install("tqdm")
    .pip_install("boto3")
    .run_commands(
        "cd /root && git clone https://github.com/MeiGen-AI/MultiTalk.git",
    )
    .env({
        "PYTHONPATH": "/root/MultiTalk",
    })
)

# Test which layer to build - modify this to test different layers
TEST_LAYER = 5  # Change this to test different layers

if TEST_LAYER == 1:
    # Layer 1: xformers
    print("Testing Layer 1: xformers")
    test_image = BASE_IMAGE.pip_install(
        "xformers==0.0.28",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    test_packages = ["torch", "xformers"]

elif TEST_LAYER == 2:
    # Layer 2: xformers + core ML
    print("Testing Layer 2: xformers + core ML packages")
    test_image = (
        BASE_IMAGE
        .pip_install(
            "xformers==0.0.28",
            index_url="https://download.pytorch.org/whl/cu121",
        )
        .pip_install(
            "peft",
            "accelerate",
            "einops",
            "omegaconf",
        )
    )
    test_packages = ["xformers", "peft", "accelerate", "einops", "omegaconf"]

elif TEST_LAYER == 3:
    # Layer 3: Previous + audio
    print("Testing Layer 3: Previous + audio packages")
    test_image = (
        BASE_IMAGE
        .pip_install(
            "xformers==0.0.28",
            index_url="https://download.pytorch.org/whl/cu121",
        )
        .pip_install(
            "peft",
            "accelerate",
            "einops",
            "omegaconf",
        )
        .pip_install(
            "librosa",
            "soundfile",
            "scipy",
        )
    )
    test_packages = ["librosa", "soundfile", "scipy"]

elif TEST_LAYER == 4:
    # Layer 4: Previous + video
    print("Testing Layer 4: Previous + video packages")
    test_image = (
        BASE_IMAGE
        .pip_install(
            "xformers==0.0.28",
            index_url="https://download.pytorch.org/whl/cu121",
        )
        .pip_install(
            "peft",
            "accelerate",
            "einops",
            "omegaconf",
        )
        .pip_install(
            "librosa",
            "soundfile",
            "scipy",
        )
        .apt_install([
            "libsm6",
            "libxext6",
            "libxrender-dev",
            "libgomp1",
        ])
        .pip_install(
            "opencv-python",
            "moviepy",
            "imageio",
            "imageio-ffmpeg",
        )
    )
    test_packages = ["cv2", "moviepy", "imageio"]

elif TEST_LAYER == 5:
    # Layer 5: Full image
    print("Testing Layer 5: Full image with all packages")
    test_image = (
        BASE_IMAGE
        .pip_install(
            "xformers==0.0.28",
            index_url="https://download.pytorch.org/whl/cu121",
        )
        .pip_install(
            "peft",
            "accelerate",
            "einops",
            "omegaconf",
        )
        .pip_install(
            "librosa",
            "soundfile",
            "scipy",
        )
        .apt_install([
            "libsm6",
            "libxext6",
            "libxrender-dev",
            "libgomp1",
        ])
        .pip_install(
            "opencv-python",
            "moviepy",
            "imageio",
            "imageio-ffmpeg",
        )
        .pip_install(
            "diffusers>=0.30.0",
            "Pillow",
            "numba==0.59.1",
            "psutil",
            "packaging",
            "ninja",
        )
    )
    test_packages = ["diffusers", "PIL", "numba", "psutil", "cv2", "moviepy"]

# Create test app
app = modal.App(f"test-layer-{TEST_LAYER}")

@app.function(
    image=test_image,
    gpu="t4",
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def test_environment():
    """Test the environment with specific packages."""
    import sys
    results = {
        "layer": TEST_LAYER,
        "python": sys.version.split()[0],
        "packages": {},
    }
    
    # Test CUDA
    try:
        import torch
        results["cuda_available"] = bool(torch.cuda.is_available())
        if results["cuda_available"]:
            results["gpu_name"] = str(torch.cuda.get_device_name(0))
            # Quick GPU test
            test_tensor = torch.randn(100, 100).cuda()
            results["gpu_compute"] = "pass"
    except Exception as e:
        results["cuda_error"] = str(e)[:100]
    
    # Test packages
    for package in test_packages:
        try:
            if package == "cv2":
                import cv2
                results["packages"][package] = str(cv2.__version__)
            elif package == "PIL":
                import PIL
                results["packages"][package] = str(PIL.__version__)
            else:
                module = __import__(package)
                version = getattr(module, "__version__", "installed")
                results["packages"][package] = str(version)
        except ImportError:
            results["packages"][package] = "NOT_FOUND"
        except Exception as e:
            results["packages"][package] = f"ERROR: {str(e)[:50]}"
    
    # Special tests for specific layers
    if TEST_LAYER >= 1:
        # Test xformers memory efficient attention
        try:
            import xformers.ops
            q = torch.randn(1, 8, 128, 64).cuda()
            k = torch.randn(1, 8, 128, 64).cuda()
            v = torch.randn(1, 8, 128, 64).cuda()
            out = xformers.ops.memory_efficient_attention(q, k, v)
            results["xformers_attention"] = "working"
        except Exception as e:
            results["xformers_attention"] = f"failed: {str(e)[:50]}"
    
    return results

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"Testing Layer {TEST_LAYER}")
    print(f"{'='*60}\n")
    
    with app.run():
        result = test_environment.remote()
        
        print(f"Python: {result['python']}")
        print(f"CUDA: {result.get('cuda_available', 'error')}")
        if "gpu_name" in result:
            print(f"GPU: {result['gpu_name']}")
        
        print("\nPackages:")
        all_good = True
        for pkg, status in result["packages"].items():
            if status == "NOT_FOUND" or status.startswith("ERROR"):
                print(f"  ❌ {pkg}: {status}")
                all_good = False
            else:
                print(f"  ✅ {pkg}: {status}")
        
        # Additional test results
        if "xformers_attention" in result:
            if result["xformers_attention"] == "working":
                print(f"\n✅ xformers attention: {result['xformers_attention']}")
            else:
                print(f"\n❌ xformers attention: {result['xformers_attention']}")
                all_good = False
        
        print(f"\n{'='*60}")
        if all_good and result.get("cuda_available"):
            print(f"✅ LAYER {TEST_LAYER} TEST PASSED!")
            print(f"\nTo test the next layer, change TEST_LAYER to {TEST_LAYER + 1}")
        else:
            print(f"❌ LAYER {TEST_LAYER} TEST FAILED!")
            print("\nFix the issues before proceeding to the next layer")
        print(f"{'='*60}")