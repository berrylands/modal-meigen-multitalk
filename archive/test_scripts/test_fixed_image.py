"""
Test the fixed Modal image with inline definition.
"""

import modal
import os
import sys

# Enable full debugging
sys.stdout.reconfigure(line_buffering=True)
modal.enable_output()
os.environ["MODAL_LOGLEVEL"] = "DEBUG"

# Set up Modal authentication
if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

# Define image inline to avoid import issues
multitalk_image = (
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
        "PYTHONPATH": "/root/MultiTalk",  # Fixed - no shell expansion
    })
)

app = modal.App("test-fixed-image")

@app.function(
    image=multitalk_image,
    gpu="t4",
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("aws-secret"),
    ],
)
def test_complete_environment():
    """Test the complete fixed environment."""
    import torch
    import transformers
    import os
    import sys
    
    results = {
        "status": "testing",
        "python_version": sys.version.split()[0],
    }
    
    # Test PyTorch
    try:
        results["torch_version"] = str(torch.__version__)
        results["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            results["gpu_name"] = str(torch.cuda.get_device_name(0))
            results["cuda_version"] = str(torch.version.cuda)
            
            # Quick GPU test
            test_tensor = torch.randn(100, 100).cuda()
            results["gpu_test"] = "pass"
    except Exception as e:
        results["torch_error"] = str(e)
    
    # Test transformers
    try:
        results["transformers_version"] = str(transformers.__version__)
    except Exception as e:
        results["transformers_error"] = str(e)
    
    # Test MultiTalk repo
    results["multitalk_exists"] = os.path.exists("/root/MultiTalk")
    if results["multitalk_exists"]:
        # Check key files
        for file in ["generate_multitalk.py", "app.py", "requirements.txt"]:
            path = f"/root/MultiTalk/{file}"
            results[f"multitalk_{file}"] = os.path.exists(path)
    
    # Test environment
    results["pythonpath"] = os.environ.get("PYTHONPATH", "Not set")
    results["hf_token"] = "HF_TOKEN" in os.environ or "HUGGINGFACE_TOKEN" in os.environ
    results["aws_configured"] = "AWS_ACCESS_KEY_ID" in os.environ
    
    # Overall status
    critical_checks = [
        results.get("cuda_available", False),
        results.get("multitalk_exists", False),
        results.get("transformers_version") is not None,
        results.get("hf_token", False),
        results.get("aws_configured", False),
    ]
    
    results["status"] = "ready" if all(critical_checks) else "incomplete"
    results["passed_checks"] = sum(critical_checks)
    results["total_checks"] = len(critical_checks)
    
    return results

if __name__ == "__main__":
    print("\nTesting fixed Modal image configuration...")
    print("=" * 60)
    
    try:
        with app.run():
            print("\nRunning environment test...")
            result = test_complete_environment.remote()
            
            print("\n" + "=" * 60)
            print("Test Results:")
            print("=" * 60)
            
            for key, value in sorted(result.items()):
                print(f"{key}: {value}")
            
            print("\n" + "=" * 60)
            print(f"Status: {result['status'].upper()}")
            print(f"Checks passed: {result['passed_checks']}/{result['total_checks']}")
            print("=" * 60)
            
            if result["status"] == "ready":
                print("\n✅ Environment is ready for MeiGen-MultiTalk!")
                print("\nNext steps:")
                print("1. Test with the full image including all ML packages")
                print("2. Implement model download functionality")
                print("3. Create inference wrapper")
            else:
                print("\n⚠️  Some components are missing or not configured")
                
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {str(e)}")
        if os.environ.get("MODAL_TRACEBACK") == "1":
            raise