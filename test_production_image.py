"""
Comprehensive test of the production Modal image.
"""

import modal
import os

modal.enable_output()

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

from modal_image_production import multitalk_image

app = modal.App("test-production-image")

@app.function(
    image=multitalk_image,
    gpu="a10g",  # Use better GPU for production test
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("aws-secret"),
    ],
    timeout=300,
)
def test_production_environment():
    """Comprehensive test of production environment."""
    import torch
    import sys
    import os
    import subprocess
    
    print("="*60)
    print("MeiGen-MultiTalk Production Environment Test")
    print("="*60)
    
    results = {"tests": {}}
    
    # 1. Basic environment
    print("\n1. Basic Environment:")
    results["python_version"] = sys.version.split()[0]
    results["cuda_available"] = torch.cuda.is_available()
    print(f"   Python: {results['python_version']}")
    print(f"   CUDA: {results['cuda_available']}")
    
    if results["cuda_available"]:
        results["gpu_name"] = torch.cuda.get_device_name(0)
        results["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        print(f"   GPU: {results['gpu_name']} ({results['gpu_memory_gb']} GB)")
    
    # 2. Test all critical packages
    print("\n2. Package Versions:")
    packages = {
        "torch": None,
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
        "numba": None,
    }
    
    for pkg_name in packages:
        try:
            if pkg_name == "cv2":
                import cv2
                version = cv2.__version__
            else:
                module = __import__(pkg_name)
                version = getattr(module, "__version__", "installed")
            packages[pkg_name] = version
            print(f"   ✅ {pkg_name}: {version}")
            results["tests"][f"package_{pkg_name}"] = True
        except Exception as e:
            print(f"   ❌ {pkg_name}: {str(e)[:50]}")
            results["tests"][f"package_{pkg_name}"] = False
    
    # 3. Test MultiTalk repository
    print("\n3. MultiTalk Repository:")
    multitalk_path = "/root/MultiTalk"
    results["multitalk_exists"] = os.path.exists(multitalk_path)
    print(f"   Repository exists: {results['multitalk_exists']}")
    
    if results["multitalk_exists"]:
        key_files = {
            "generate_multitalk.py": "Main generation script",
            "app.py": "Application entry",
            "requirements.txt": "Dependencies",
            "configs": "Configuration directory",
        }
        
        for file, desc in key_files.items():
            path = os.path.join(multitalk_path, file)
            exists = os.path.exists(path)
            results["tests"][f"multitalk_{file}"] = exists
            print(f"   {'✅' if exists else '❌'} {file}: {desc}")
    
    # 4. Test xformers attention
    print("\n4. xformers Attention Test:")
    try:
        import xformers.ops
        batch_size = 2
        seq_len = 512
        n_heads = 8
        d_head = 64
        
        q = torch.randn(batch_size, n_heads, seq_len, d_head).cuda()
        k = torch.randn(batch_size, n_heads, seq_len, d_head).cuda()
        v = torch.randn(batch_size, n_heads, seq_len, d_head).cuda()
        
        out = xformers.ops.memory_efficient_attention(q, k, v)
        results["tests"]["xformers_attention"] = True
        print(f"   ✅ Memory efficient attention working")
        print(f"   Output shape: {out.shape}")
    except Exception as e:
        results["tests"]["xformers_attention"] = False
        print(f"   ❌ Failed: {str(e)[:100]}")
    
    # 5. Test model loading capabilities
    print("\n5. Model Loading Test:")
    try:
        from transformers import AutoTokenizer, AutoModel
        from diffusers import StableDiffusionPipeline
        
        # Just verify we can import and access HF
        results["tests"]["hf_access"] = "HF_TOKEN" in os.environ or "HUGGINGFACE_TOKEN" in os.environ
        print(f"   HuggingFace token: {'✅ Available' if results['tests']['hf_access'] else '❌ Missing'}")
        
        # Test we can create model instances (not load weights)
        results["tests"]["model_imports"] = True
        print("   ✅ Model imports successful")
    except Exception as e:
        results["tests"]["model_imports"] = False
        print(f"   ❌ Model import failed: {str(e)[:100]}")
    
    # 6. Test AWS access
    print("\n6. AWS Access Test:")
    try:
        import boto3
        results["tests"]["aws_configured"] = "AWS_ACCESS_KEY_ID" in os.environ
        print(f"   AWS credentials: {'✅ Configured' if results['tests']['aws_configured'] else '❌ Missing'}")
        
        if results["tests"]["aws_configured"]:
            s3 = boto3.client('s3')
            # Just verify client creation works
            results["tests"]["s3_client"] = True
            print("   ✅ S3 client created")
    except Exception as e:
        results["tests"]["s3_client"] = False
        print(f"   ❌ S3 client failed: {str(e)[:50]}")
    
    # 7. Memory and compute test
    print("\n7. GPU Memory and Compute Test:")
    try:
        # Allocate significant GPU memory
        size = 2048
        a = torch.randn(size, size).cuda()
        b = torch.randn(size, size).cuda()
        c = torch.matmul(a, b)
        
        allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        results["tests"]["gpu_compute"] = True
        print(f"   ✅ GPU compute working")
        print(f"   Memory allocated: {allocated_gb:.2f} GB")
        
        # Cleanup
        del a, b, c
        torch.cuda.empty_cache()
    except Exception as e:
        results["tests"]["gpu_compute"] = False
        print(f"   ❌ GPU compute failed: {str(e)[:50]}")
    
    # Summary
    total_tests = len(results["tests"])
    passed_tests = sum(results["tests"].values())
    results["tests_passed"] = passed_tests
    results["tests_total"] = total_tests
    results["all_passed"] = passed_tests == total_tests
    
    print("\n" + "="*60)
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print("="*60)
    
    return results

if __name__ == "__main__":
    print("Testing MeiGen-MultiTalk Production Image...")
    
    with app.run():
        result = test_production_environment.remote()
        
        if result["all_passed"]:
            print("\n✅ ALL TESTS PASSED!")
            print("\nProduction image is ready for use.")
            print("\nNext steps:")
            print("1. Implement model download functionality")
            print("2. Test actual inference pipeline")
            print("3. Deploy to production")
        else:
            print(f"\n⚠️  {result['tests_passed']}/{result['tests_total']} tests passed")
            print("\nFailed tests:")
            for test, passed in result["tests"].items():
                if not passed:
                    print(f"  - {test}")