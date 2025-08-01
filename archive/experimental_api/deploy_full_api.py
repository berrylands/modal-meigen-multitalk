#!/usr/bin/env python3
"""
Deploy the full MultiTalk API with actual video generation capabilities.
This combines the REST API with the CUDA-based video generation.
"""

import modal
import subprocess
import sys

def deploy_full_api():
    """Deploy both the MultiTalk model and the API."""
    
    print("=" * 60)
    print("Deploying Full MeiGen-MultiTalk API to Modal")
    print("=" * 60)
    
    # First, deploy the CUDA app
    print("\n1. Deploying MultiTalk CUDA model...")
    try:
        result = subprocess.run(
            ["modal", "deploy", "app_multitalk_cuda.py", "--name", "multitalk-cuda"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Error deploying CUDA app: {result.stderr}")
            return False
        print("✓ CUDA model deployed successfully")
    except Exception as e:
        print(f"Failed to deploy CUDA app: {e}")
        return False
    
    # Then deploy the API
    print("\n2. Deploying REST API...")
    try:
        result = subprocess.run(
            ["modal", "deploy", "api.py", "--name", "multitalk-api"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Error deploying API: {result.stderr}")
            return False
        print("✓ API deployed successfully")
        
        # Extract the API URL from the output
        for line in result.stdout.split('\n'):
            if "https://" in line and "modal.run" in line:
                api_url = line.strip().split()[-1]
                print(f"\n✓ API URL: {api_url}")
                break
    except Exception as e:
        print(f"Failed to deploy API: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("Deployment Complete!")
    print("=" * 60)
    print("\nThe full MultiTalk API is now deployed with:")
    print("- ✓ CUDA-based video generation model")
    print("- ✓ REST API endpoints")
    print("- ✓ S3 integration")
    print("- ✓ Async job processing")
    
    return True

if __name__ == "__main__":
    success = deploy_full_api()
    sys.exit(0 if success else 1)