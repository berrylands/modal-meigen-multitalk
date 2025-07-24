"""Final comprehensive secrets test."""

import modal
import os

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

app = modal.App("test-final-secrets")

@app.function(
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("aws-secret")
    ],
    image=modal.Image.debian_slim().pip_install(["boto3", "requests"])
)
def verify_all_secrets():
    """Comprehensive secret verification."""
    import os
    import boto3
    import requests
    
    print("Modal Secrets Final Verification")
    print("=" * 60)
    
    results = {}
    
    # 1. AWS Secret Test
    print("\n1. AWS Secret Verification:")
    aws_key_exists = "AWS_ACCESS_KEY_ID" in os.environ
    aws_secret_exists = "AWS_SECRET_ACCESS_KEY" in os.environ
    aws_region = os.environ.get("AWS_REGION", "not set")
    
    print(f"   AWS_ACCESS_KEY_ID: {'✅ exists' if aws_key_exists else '❌ missing'}")
    print(f"   AWS_SECRET_ACCESS_KEY: {'✅ exists' if aws_secret_exists else '❌ missing'}")
    print(f"   AWS_REGION: {aws_region}")
    
    if aws_key_exists and aws_secret_exists:
        try:
            s3 = boto3.client('s3', region_name=aws_region)
            buckets = s3.list_buckets()
            bucket_count = len(buckets.get('Buckets', []))
            print(f"   ✅ S3 Access: Working ({bucket_count} buckets found)")
            results['aws_s3_access'] = True
        except Exception as e:
            print(f"   ❌ S3 Access: Failed - {str(e)[:50]}...")
            results['aws_s3_access'] = False
    
    # 2. HuggingFace Secret Test
    print("\n2. HuggingFace Secret Verification:")
    
    # Check various possible env var names
    hf_vars = ["HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_TOKEN"]
    hf_token = None
    hf_var_name = None
    
    for var in hf_vars:
        if var in os.environ:
            hf_token = os.environ[var]
            hf_var_name = var
            break
    
    if hf_token:
        print(f"   ✅ Token found as: {hf_var_name}")
        print(f"   Token length: {len(hf_token)} characters")
        print(f"   Token prefix: {hf_token[:8]}...")
        
        # Test HF API access
        try:
            headers = {"Authorization": f"Bearer {hf_token}"}
            response = requests.get(
                "https://huggingface.co/api/whoami",
                headers=headers,
                timeout=10
            )
            if response.status_code == 200:
                user_data = response.json()
                print(f"   ✅ HF API Access: Authenticated as '{user_data.get('name', 'unknown')}'")
                results['hf_api_access'] = True
                
                # Set standard env var for compatibility
                if hf_var_name != "HUGGINGFACE_TOKEN":
                    os.environ["HUGGINGFACE_TOKEN"] = hf_token
                    print(f"   ℹ️  Set HUGGINGFACE_TOKEN from {hf_var_name}")
            else:
                print(f"   ❌ HF API Access: Failed with status {response.status_code}")
                print(f"   Response: {response.text[:100]}...")
                results['hf_api_access'] = False
        except Exception as e:
            print(f"   ❌ HF API Access: Error - {str(e)}")
            results['hf_api_access'] = False
    else:
        print("   ❌ No HuggingFace token found in any expected variable")
        results['hf_api_access'] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  AWS S3 Access: {'✅ Working' if results.get('aws_s3_access') else '❌ Failed'}")
    print(f"  HuggingFace API: {'✅ Working' if results.get('hf_api_access') else '❌ Failed'}")
    
    all_good = all(results.values()) if results else False
    if all_good:
        print("\n🎉 All secrets are properly configured and working!")
    else:
        print("\n⚠️  Some secrets need attention")
    
    return results

if __name__ == "__main__":
    import subprocess
    subprocess.run(["modal", "run", __file__ + "::verify_all_secrets"])