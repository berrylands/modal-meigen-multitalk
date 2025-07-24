"""Test Modal authentication and basic functionality."""

import modal
import sys

app = modal.App("test-auth")

@app.function()
def test_basic():
    """Test basic Modal function."""
    return "Modal is working!"

@app.function(secrets=[
    modal.Secret.from_name("huggingface-secret"),
    modal.Secret.from_name("aws-secret")
])
def test_secrets():
    """Test secret access."""
    import os
    results = {
        "huggingface_token": "HUGGINGFACE_TOKEN" in os.environ,
        "aws_access_key": "AWS_ACCESS_KEY_ID" in os.environ,
        "aws_secret_key": "AWS_SECRET_ACCESS_KEY" in os.environ,
        "aws_region": os.environ.get("AWS_REGION", "not set")
    }
    return results

@app.function(gpu="t4")
def test_gpu():
    """Test GPU availability."""
    import torch
    return {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"
    }

@app.local_entrypoint()
def main():
    """Run all tests."""
    print("Testing Modal setup...\n")
    
    # Test basic function
    print("1. Testing basic function:")
    try:
        result = test_basic.remote()
        print(f"   ✓ {result}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        print("\n   Please run 'modal setup' to authenticate")
        sys.exit(1)
    
    # Test secrets
    print("\n2. Testing secrets access:")
    try:
        results = test_secrets.remote()
        for key, value in results.items():
            status = "✓" if value and key != "aws_region" else "✗"
            print(f"   {status} {key}: {value}")
        if not all(v for k, v in results.items() if k != "aws_region"):
            print("\n   Note: Create secrets using the Modal dashboard or CLI")
    except Exception as e:
        print(f"   ✗ Error accessing secrets: {e}")
    
    # Test GPU
    print("\n3. Testing GPU access:")
    try:
        gpu_info = test_gpu.remote()
        for key, value in gpu_info.items():
            print(f"   • {key}: {value}")
    except Exception as e:
        print(f"   ✗ Error accessing GPU: {e}")
    
    print("\n✅ Modal setup test complete!")

if __name__ == "__main__":
    main()