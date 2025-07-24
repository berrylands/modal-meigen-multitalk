"""Simple AWS secret test for Modal."""

import modal
import os

# Set auth token if running locally
if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

app = modal.App("test-aws-simple")

@app.function(
    secrets=[modal.Secret.from_name("aws-secret")],
    image=modal.Image.debian_slim().pip_install("boto3")
)
def check_aws():
    """Check AWS credentials."""
    import os
    print("AWS Secret Test")
    print("=" * 40)
    print(f"AWS_ACCESS_KEY_ID exists: {'AWS_ACCESS_KEY_ID' in os.environ}")
    print(f"AWS_SECRET_ACCESS_KEY exists: {'AWS_SECRET_ACCESS_KEY' in os.environ}")
    print(f"AWS_REGION: {os.environ.get('AWS_REGION', 'not set')}")
    
    if "AWS_ACCESS_KEY_ID" in os.environ:
        print(f"Key prefix: {os.environ['AWS_ACCESS_KEY_ID'][:4]}...")
    
    # Test S3 access
    try:
        import boto3
        s3 = boto3.client('s3')
        response = s3.list_buckets()
        print(f"\n✅ S3 Access successful! Found {len(response['Buckets'])} buckets")
        return True
    except Exception as e:
        print(f"\n❌ S3 Access failed: {e}")
        return False

if __name__ == "__main__":
    # For local testing
    import subprocess
    import sys
    
    # Run with modal
    cmd = ["modal", "run", __file__]
    subprocess.run(cmd)