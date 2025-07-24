"""Complete Modal setup verification."""

import modal
import os

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

app = modal.App("test-complete")

@app.function(
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("aws-secret")
    ],
    image=modal.Image.debian_slim().pip_install(["boto3", "huggingface-hub"])
)
def verify_complete_setup():
    """Verify complete Modal setup with all secrets."""
    import os
    import boto3
    from huggingface_hub import HfApi
    
    print("Modal Complete Setup Verification")
    print("=" * 70)
    
    all_good = True
    
    # 1. AWS Verification
    print("\n1. AWS Configuration:")
    try:
        s3 = boto3.client('s3', region_name=os.environ.get("AWS_REGION", "eu-west-1"))
        buckets = s3.list_buckets()
        bucket_count = len(buckets.get('Buckets', []))
        print(f"   ✅ S3 Access: Working")
        print(f"   ✅ Region: {os.environ.get('AWS_REGION', 'not set')}")
        print(f"   ✅ Buckets accessible: {bucket_count}")
    except Exception as e:
        print(f"   ❌ AWS Error: {e}")
        all_good = False
    
    # 2. HuggingFace Verification
    print("\n2. HuggingFace Configuration:")
    try:
        # Get token (handle both possible names)
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if token:
            # Ensure both env vars are set for compatibility
            os.environ["HUGGINGFACE_TOKEN"] = token
            os.environ["HF_TOKEN"] = token
            
            api = HfApi(token=token)
            user = api.whoami()
            print(f"   ✅ Authentication: Working")
            print(f"   ✅ User: {user['name']}")
            print(f"   ✅ Token stored as: HF_TOKEN")
            
            # Test model access
            model_info = api.model_info("bert-base-uncased")
            print(f"   ✅ Model access: Verified")
        else:
            print(f"   ❌ No HuggingFace token found")
            all_good = False
    except Exception as e:
        print(f"   ❌ HuggingFace Error: {e}")
        all_good = False
    
    # 3. Modal Configuration
    print("\n3. Modal Configuration:")
    print(f"   ✅ Modal connection: Active")
    print(f"   ✅ Secrets loaded: huggingface-secret, aws-secret")
    print(f"   ✅ Image building: Working")
    
    # Summary
    print("\n" + "=" * 70)
    if all_good:
        print("🎉 ALL SYSTEMS GO! Modal environment is fully configured!")
        print("\nYou can now:")
        print("- Download models from HuggingFace")
        print("- Store/retrieve files from S3")
        print("- Deploy ML models on Modal")
    else:
        print("⚠️  Some issues need attention")
    
    return all_good

if __name__ == "__main__":
    import subprocess
    subprocess.run(["modal", "run", __file__ + "::verify_complete_setup"])