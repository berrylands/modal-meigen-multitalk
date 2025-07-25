"""
Incremental Modal image builder for MeiGen-MultiTalk.
Tests each layer of dependencies before adding the next.
"""

import modal
import os
import sys
from typing import Dict, List, Tuple

# Enable debugging
sys.stdout.reconfigure(line_buffering=True)
modal.enable_output()
os.environ["MODAL_LOGLEVEL"] = "DEBUG"

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

def test_image(image: modal.Image, name: str, test_packages: List[str]) -> Tuple[bool, Dict]:
    """Test an image with specific package checks."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    app = modal.App(f"test-{name.lower().replace(' ', '-')}")
    
    @app.function(
        image=image,
        gpu="t4",
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
    def test_env():
        import sys
        import os
        results = {
            "python": sys.version.split()[0],
            "cuda_available": False,
            "packages": {},
        }
        
        # Test CUDA
        try:
            import torch
            results["cuda_available"] = bool(torch.cuda.is_available())
            if results["cuda_available"]:
                results["gpu_name"] = str(torch.cuda.get_device_name(0))
        except:
            pass
        
        # Test requested packages
        for package in test_packages:
            try:
                if package == "cv2":
                    import cv2
                    results["packages"][package] = str(cv2.__version__)
                elif package == "flash_attn":
                    import flash_attn
                    results["packages"][package] = str(flash_attn.__version__)
                else:
                    module = __import__(package)
                    version = getattr(module, "__version__", "installed")
                    results["packages"][package] = str(version)
            except ImportError:
                results["packages"][package] = "FAILED"
            except Exception as e:
                results["packages"][package] = f"ERROR: {str(e)[:50]}"
        
        return results
    
    try:
        with app.run():
            result = test_env.remote()
            
            # Print results
            print(f"Python: {result['python']}")
            print(f"CUDA: {result['cuda_available']}")
            if "gpu_name" in result:
                print(f"GPU: {result['gpu_name']}")
            
            print("\nPackages:")
            all_good = True
            for pkg, status in result["packages"].items():
                if status == "FAILED" or status.startswith("ERROR"):
                    print(f"  ❌ {pkg}: {status}")
                    all_good = False
                else:
                    print(f"  ✅ {pkg}: {status}")
            
            success = all_good and result["cuda_available"]
            print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")
            
            return success, result
            
    except Exception as e:
        print(f"❌ Build/Test Failed: {str(e)[:200]}...")
        return False, {"error": str(e)}

def main():
    """Incrementally build and test the Modal image."""
    print("MeiGen-MultiTalk Incremental Image Builder")
    print("="*60)
    
    current_image = BASE_IMAGE
    all_layers = []
    
    # Layer 1: xformers (alternative to flash-attn)
    print("\n\n🔨 LAYER 1: Adding xformers...")
    layer1_image = current_image.pip_install(
        "xformers==0.0.28",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    
    success, result = test_image(layer1_image, "Layer 1 - xformers", ["xformers"])
    if success:
        current_image = layer1_image
        all_layers.append("xformers")
        print("✅ Layer 1 added successfully")
    else:
        print("❌ Layer 1 failed - stopping here")
        return
    
    # Layer 2: Core ML packages
    print("\n\n🔨 LAYER 2: Adding core ML packages...")
    layer2_image = current_image.pip_install(
        "peft",
        "accelerate",
        "einops",
        "omegaconf",
    )
    
    success, result = test_image(
        layer2_image, 
        "Layer 2 - Core ML", 
        ["peft", "accelerate", "einops", "omegaconf"]
    )
    if success:
        current_image = layer2_image
        all_layers.append("core ML packages")
        print("✅ Layer 2 added successfully")
    else:
        print("❌ Layer 2 failed")
        return
    
    # Layer 3: Audio packages
    print("\n\n🔨 LAYER 3: Adding audio packages...")
    layer3_image = current_image.pip_install(
        "librosa",
        "soundfile",
        "scipy",
    )
    
    success, result = test_image(
        layer3_image,
        "Layer 3 - Audio",
        ["librosa", "soundfile", "scipy"]
    )
    if success:
        current_image = layer3_image
        all_layers.append("audio packages")
        print("✅ Layer 3 added successfully")
    else:
        print("❌ Layer 3 failed")
        return
    
    # Layer 4: Video packages (with system deps)
    print("\n\n🔨 LAYER 4: Adding video packages...")
    layer4_image = (
        current_image
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
    
    success, result = test_image(
        layer4_image,
        "Layer 4 - Video",
        ["cv2", "moviepy", "imageio"]
    )
    if success:
        current_image = layer4_image
        all_layers.append("video packages")
        print("✅ Layer 4 added successfully")
    else:
        print("❌ Layer 4 failed")
        return
    
    # Layer 5: Diffusers and remaining packages
    print("\n\n🔨 LAYER 5: Adding diffusers and remaining packages...")
    layer5_image = current_image.pip_install(
        "diffusers>=0.30.0",
        "Pillow",
        "numba==0.59.1",
        "psutil",
        "packaging",
        "ninja",
    )
    
    success, result = test_image(
        layer5_image,
        "Layer 5 - Diffusers",
        ["diffusers", "PIL", "numba", "psutil"]
    )
    if success:
        current_image = layer5_image
        all_layers.append("diffusers and utilities")
        print("✅ Layer 5 added successfully")
    else:
        print("❌ Layer 5 failed")
    
    # Final summary
    print("\n\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Successfully added {len(all_layers)} layers:")
    for i, layer in enumerate(all_layers, 1):
        print(f"{i}. {layer}")
    
    if len(all_layers) == 5:
        print("\n✅ ALL LAYERS BUILT SUCCESSFULLY!")
        print("\nNext steps:")
        print("1. Save this as the production image")
        print("2. Test actual inference")
        print("3. Consider adding flash-attn if needed")
    else:
        print(f"\n⚠️  Only {len(all_layers)}/5 layers built successfully")
        print("Debug the failing layer before proceeding")

if __name__ == "__main__":
    main()