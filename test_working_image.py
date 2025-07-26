"""
Test the working Modal image configuration.
"""

import modal
import os

modal.enable_output()

# Set up Modal authentication
if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

from modal_image_working import multitalk_image

app = modal.App("test-working-image")

@app.function(
    image=multitalk_image,
    gpu="t4",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=300,
)
def test_environment():
    """Comprehensive test of the working environment."""
    import sys
    import subprocess
    import os
    
    print("=" * 60)
    print("MeiGen-MultiTalk Environment Test")
    print("=" * 60)
    
    results = {"status": "testing"}
    
    # 1. Python version
    results["python_version"] = sys.version.split()[0]
    print(f"\n1. Python: {results['python_version']}")
    
    # 2. CUDA/PyTorch
    print("\n2. PyTorch/CUDA:")
    try:
        import torch
        results["torch_version"] = torch.__version__
        results["cuda_available"] = torch.cuda.is_available()
        print(f"   PyTorch: {results['torch_version']}")
        print(f"   CUDA available: {results['cuda_available']}")
        if torch.cuda.is_available():
            results["cuda_version"] = torch.version.cuda
            results["gpu_name"] = torch.cuda.get_device_name(0)
            results["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            print(f"   CUDA version: {results['cuda_version']}")
            print(f"   GPU: {results['gpu_name']} ({results['gpu_memory_gb']} GB)")
    except Exception as e:
        results["torch_error"] = str(e)
        print(f"   ❌ Error: {e}")
    
    # 3. Key ML packages
    print("\n3. ML Packages:")
    packages = {
        "transformers": None,
        "diffusers": None,
        "xformers": None,
        "accelerate": None,
        "peft": None,
        "librosa": None,
        "moviepy": None,
        "cv2": None,
        "einops": None,
        "omegaconf": None,
    }
    
    for package in packages:
        try:
            if package == "cv2":
                import cv2
                version = cv2.__version__
            else:
                module = __import__(package)
                version = getattr(module, "__version__", "installed")
            packages[package] = version
            print(f"   ✅ {package}: {version}")
        except ImportError as e:
            packages[package] = "not found"
            print(f"   ❌ {package}: not found")
    
    results["packages"] = packages
    
    # 4. System tools
    print("\n4. System Tools:")
    for tool in ["git", "ffmpeg", "ninja"]:
        try:
            result = subprocess.run([tool, "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                version = result.stdout.split('\n')[0]
                print(f"   ✅ {tool}: {version}")
                results[f"{tool}_available"] = True
            else:
                print(f"   ❌ {tool}: error")
                results[f"{tool}_available"] = False
        except:
            print(f"   ❌ {tool}: not found")
            results[f"{tool}_available"] = False
    
    # 5. MultiTalk repository
    print("\n5. MultiTalk Repository:")
    multitalk_path = "/root/MultiTalk"
    if os.path.exists(multitalk_path):
        print(f"   ✅ Repository exists at {multitalk_path}")
        results["multitalk_exists"] = True
        
        # Check key files
        key_files = ["generate_multitalk.py", "app.py", "requirements.txt"]
        for file in key_files:
            path = os.path.join(multitalk_path, file)
            exists = os.path.exists(path)
            results[f"multitalk_{file}"] = exists
            print(f"   {'✅' if exists else '❌'} {file}: {'found' if exists else 'missing'}")
    else:
        print(f"   ❌ Repository not found at {multitalk_path}")
        results["multitalk_exists"] = False
    
    # 6. Environment variables
    print("\n6. Environment:")
    env_vars = {
        "PYTHONPATH": os.environ.get("PYTHONPATH", "Not set"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"),
        "HF_TOKEN": "Set" if "HF_TOKEN" in os.environ else "Not set",
        "HUGGINGFACE_TOKEN": "Set" if "HUGGINGFACE_TOKEN" in os.environ else "Not set",
    }
    
    for var, value in env_vars.items():
        print(f"   {var}: {value}")
        results[f"env_{var}"] = value != "Not set"
    
    # 7. Test attention mechanism
    print("\n7. Attention Optimization:")
    # Check if we have xformers (alternative to flash-attn)
    try:
        import xformers
        import xformers.ops
        print(f"   ✅ xformers available: {xformers.__version__}")
        results["xformers_available"] = True
        
        # Test xformers memory efficient attention
        try:
            import torch
            if torch.cuda.is_available():
                # Small test of memory efficient attention
                q = torch.randn(1, 8, 128, 64).cuda()
                k = torch.randn(1, 8, 128, 64).cuda()
                v = torch.randn(1, 8, 128, 64).cuda()
                out = xformers.ops.memory_efficient_attention(q, k, v)
                print("   ✅ Memory efficient attention working")
                results["xformers_attention_works"] = True
        except Exception as e:
            print(f"   ⚠️  Memory efficient attention test failed: {e}")
            results["xformers_attention_works"] = False
    except:
        print("   ⚠️  xformers not available (but not critical)")
        results["xformers_available"] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    
    critical_checks = [
        ("CUDA available", results.get("cuda_available", False)),
        ("MultiTalk repo", results.get("multitalk_exists", False)),
        ("PyTorch installed", "torch_version" in results),
        ("Transformers installed", packages.get("transformers") != "not found"),
        ("HF token available", results.get("env_HF_TOKEN", False) or results.get("env_HUGGINGFACE_TOKEN", False)),
    ]
    
    all_good = True
    for check, passed in critical_checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")
        if not passed:
            all_good = False
    
    results["status"] = "ready" if all_good else "incomplete"
    
    print("=" * 60)
    return results

if __name__ == "__main__":
    print("Testing working Modal image configuration...")
    
    with app.run():
        try:
            result = test_environment.remote()
            
            print(f"\nOverall status: {result['status']}")
            
            if result['status'] == 'ready':
                print("\n✅ Environment is ready for MeiGen-MultiTalk!")
                print("\nNext steps:")
                print("1. Implement model download functionality")
                print("2. Create inference wrapper")
                print("3. Test video generation")
            else:
                print("\n⚠️  Some components are missing or not configured")
                print("Please check the output above for details")
                
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            print("\nThis likely means the image build failed.")
            print("Check the error message for details.")