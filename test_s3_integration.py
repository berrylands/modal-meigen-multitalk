"""
Test S3 integration with Modal.
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

app = modal.App("test-s3-integration")

# Mount the s3_utils.py file
s3_utils_mount = modal.mount.from_local_file(
    local_path="s3_utils.py",
    remote_path="/root/s3_utils.py"
)

@app.function(
    image=test_image,
    secrets=[modal.Secret.from_name("aws-secret")],
    mounts=[s3_utils_mount],
    timeout=300,
)
def test_s3_operations(bucket_name: str, image_key: str = "multi1.png", audio_key: str = "1.wav"):
    """Test S3 download and upload operations."""
    import sys
    sys.path.insert(0, '/root')
    
    from s3_utils import S3Manager
    import tempfile
    import os
    
    print("="*60)
    print("S3 Integration Test")
    print("="*60)
    
    results = {
        "bucket": bucket_name,
        "image_key": image_key,
        "audio_key": audio_key,
        "success": False,
        "errors": []
    }
    
    try:
        # Initialize S3 manager
        print(f"\n1. Initializing S3 manager for bucket: {bucket_name}")
        s3_manager = S3Manager(bucket_name)
        print("   ✅ S3 manager initialized")
        
        # List bucket contents
        print("\n2. Listing bucket contents...")
        contents = s3_manager.list_bucket_contents()
        print(f"   Found {len(contents)} objects:")
        for obj in contents[:10]:  # Show first 10
            print(f"   - {obj}")
        if len(contents) > 10:
            print(f"   ... and {len(contents) - 10} more")
        
        # Check if our files exist
        if image_key not in contents:
            print(f"   ⚠️  Warning: {image_key} not found in bucket")
        if audio_key not in contents:
            print(f"   ⚠️  Warning: {audio_key} not found in bucket")
        
        # Download test files
        print(f"\n3. Downloading input files...")
        try:
            inputs = s3_manager.download_inputs(image_key, audio_key)
            print(f"   ✅ Downloaded image to: {inputs['image_path']}")
            print(f"   ✅ Downloaded audio to: {inputs['audio_path']}")
            
            # Verify files exist and have content
            image_size = os.path.getsize(inputs['image_path'])
            audio_size = os.path.getsize(inputs['audio_path'])
            print(f"   Image size: {image_size:,} bytes")
            print(f"   Audio size: {audio_size:,} bytes")
            
            results["download_success"] = True
            results["image_size"] = image_size
            results["audio_size"] = audio_size
            
        except Exception as e:
            print(f"   ❌ Download failed: {e}")
            results["errors"].append(f"Download: {str(e)}")
            results["download_success"] = False
        
        # Test upload with a dummy file
        print("\n4. Testing upload...")
        try:
            # Create a test file
            test_file = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
            test_file.write(b"Test upload from Modal MultiTalk")
            test_file.close()
            
            # Upload it
            s3_uri = s3_manager.upload_output(test_file.name, "test_outputs/")
            print(f"   ✅ Uploaded test file to: {s3_uri}")
            
            results["upload_success"] = True
            results["uploaded_uri"] = s3_uri
            
            # Clean up
            os.unlink(test_file.name)
            
        except Exception as e:
            print(f"   ❌ Upload failed: {e}")
            results["errors"].append(f"Upload: {str(e)}")
            results["upload_success"] = False
        
        # Overall success
        results["success"] = results.get("download_success", False) and results.get("upload_success", False)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        results["errors"].append(f"General: {str(e)}")
        results["success"] = False
    
    print("\n" + "="*60)
    if results["success"]:
        print("✅ S3 INTEGRATION TEST PASSED!")
        print("\nYou can now use S3 for:")
        print("- Downloading input images and audio")
        print("- Uploading generated videos")
    else:
        print("❌ S3 INTEGRATION TEST FAILED!")
        print(f"Errors: {results['errors']}")
    print("="*60)
    
    return results


@app.local_entrypoint()
def main():
    """Test S3 integration."""
    import sys
    
    # Get bucket name from command line or environment
    if len(sys.argv) > 1:
        bucket_name = sys.argv[1]
    else:
        bucket_name = os.environ.get('S3_BUCKET_NAME')
        
    if not bucket_name:
        print("Error: Please provide bucket name as argument or set S3_BUCKET_NAME env var")
        print("Usage: python test_s3_integration.py <bucket-name>")
        sys.exit(1)
    
    print(f"Testing S3 integration with bucket: {bucket_name}")
    print("Looking for files: multi1.png and 1.wav")
    
    result = test_s3_operations.remote(bucket_name)
    
    if result["success"]:
        print("\n✅ S3 integration is working!")
        print(f"Successfully downloaded {result.get('image_size', 0):,} + {result.get('audio_size', 0):,} bytes")
        print(f"Successfully uploaded to: {result.get('uploaded_uri', 'N/A')}")
    else:
        print("\n❌ S3 integration failed!")
        for error in result.get("errors", []):
            print(f"  - {error}")


if __name__ == "__main__":
    main()