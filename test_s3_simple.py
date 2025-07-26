"""
Simple S3 test with embedded utilities.
"""

import modal
import os

modal.enable_output()

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

# Simple test image with boto3
test_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("boto3")
)

app = modal.App("test-s3-simple")

@app.function(
    image=test_image,
    secrets=[modal.Secret.from_name("aws-secret")],
)
def test_s3_simple(bucket_name: str):
    """Simple S3 test."""
    import boto3
    
    print(f"Testing S3 access to bucket: {bucket_name}")
    
    # Create S3 client
    s3 = boto3.client('s3')
    
    # List objects
    try:
        response = s3.list_objects_v2(Bucket=bucket_name)
        
        if 'Contents' in response:
            print(f"\nFound {len(response['Contents'])} objects:")
            for obj in response['Contents']:
                print(f"  - {obj['Key']} ({obj['Size']:,} bytes)")
        else:
            print("No objects found in bucket")
        
        # Check for specific files
        keys = [obj['Key'] for obj in response.get('Contents', [])]
        has_image = 'multi1.png' in keys
        has_audio = '1.wav' in keys
        
        print(f"\nmulti1.png: {'✅ Found' if has_image else '❌ Not found'}")
        print(f"1.wav: {'✅ Found' if has_audio else '❌ Not found'}")
        
        # Try to download if found
        if has_image:
            print("\nDownloading multi1.png...")
            s3.download_file(bucket_name, 'multi1.png', '/tmp/multi1.png')
            size = os.path.getsize('/tmp/multi1.png')
            print(f"✅ Downloaded successfully ({size:,} bytes)")
        
        return {"success": True, "has_files": has_image and has_audio}
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        bucket_name = sys.argv[1]
    else:
        print("Usage: python test_s3_simple.py <bucket-name>")
        sys.exit(1)
    
    with app.run():
        result = test_s3_simple.remote(bucket_name)
        
        if result["success"]:
            print("\n✅ S3 access working!")
        else:
            print(f"\n❌ S3 access failed: {result.get('error')}")