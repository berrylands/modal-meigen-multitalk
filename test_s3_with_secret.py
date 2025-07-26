"""
Test S3 integration using bucket name from Modal secret.
"""

import modal
import os

modal.enable_output()

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

test_image = modal.Image.debian_slim(python_version="3.10").pip_install("boto3")
app = modal.App("test-s3-secret")

@app.function(
    image=test_image,
    secrets=[modal.Secret.from_name("aws-secret")],
)
def test_s3_from_secret():
    """Test S3 using bucket name from Modal secret."""
    import boto3
    import os
    
    print("="*60)
    print("S3 Test Using Modal Secret")
    print("="*60)
    
    # Get bucket name from Modal secret
    bucket_name = os.environ.get('AWS_BUCKET_NAME')
    
    if not bucket_name:
        print("❌ AWS_BUCKET_NAME not found in Modal secrets")
        return {"success": False, "error": "AWS_BUCKET_NAME not set"}
    
    print(f"\n✅ Found bucket name from secret: {bucket_name}")
    
    # Create S3 client
    s3 = boto3.client('s3')
    
    try:
        # List objects in the bucket
        print(f"\nListing objects in bucket: {bucket_name}")
        response = s3.list_objects_v2(Bucket=bucket_name, MaxKeys=20)
        
        if 'Contents' in response:
            objects = response['Contents']
            print(f"\nFound {len(objects)} objects (showing first 20):")
            
            target_files = {'multi1.png': False, '1.wav': False}
            
            for obj in objects:
                key = obj['Key']
                size = obj['Size']
                print(f"  - {key} ({size:,} bytes)")
                
                # Check for our target files
                if key in target_files:
                    target_files[key] = True
            
            # Report on target files
            print(f"\nTarget files:")
            print(f"  multi1.png: {'✅ Found' if target_files['multi1.png'] else '❌ Not found'}")
            print(f"  1.wav: {'✅ Found' if target_files['1.wav'] else '❌ Not found'}")
            
            # If both files found, try downloading one
            if all(target_files.values()):
                print("\n✅ Both target files found! Testing download...")
                
                # Download multi1.png as a test
                local_path = '/tmp/test_multi1.png'
                s3.download_file(bucket_name, 'multi1.png', local_path)
                
                file_size = os.path.getsize(local_path)
                print(f"✅ Successfully downloaded multi1.png ({file_size:,} bytes)")
                
                # Test upload
                print("\nTesting upload...")
                test_key = 'test_outputs/modal_test.txt'
                test_content = b'Modal S3 integration test successful!'
                
                s3.put_object(
                    Bucket=bucket_name,
                    Key=test_key,
                    Body=test_content
                )
                print(f"✅ Successfully uploaded test file to {test_key}")
                
                return {
                    "success": True,
                    "bucket": bucket_name,
                    "has_files": True,
                    "download_tested": True,
                    "upload_tested": True
                }
            else:
                return {
                    "success": True,
                    "bucket": bucket_name,
                    "has_files": False,
                    "missing_files": [k for k, v in target_files.items() if not v]
                }
        else:
            print("No objects found in bucket")
            return {
                "success": True,
                "bucket": bucket_name,
                "has_files": False,
                "empty": True
            }
            
    except Exception as e:
        print(f"\n❌ Error accessing bucket: {e}")
        return {
            "success": False,
            "bucket": bucket_name,
            "error": str(e)
        }

if __name__ == "__main__":
    with app.run():
        print("Testing S3 access using Modal secret AWS_BUCKET_NAME...\n")
        
        result = test_s3_from_secret.remote()
        
        print("\n" + "="*60)
        if result["success"]:
            print("✅ S3 ACCESS SUCCESSFUL!")
            print(f"Bucket: {result['bucket']}")
            
            if result.get("has_files"):
                print("\n✅ Ready for MultiTalk processing!")
                print("  - multi1.png found")
                print("  - 1.wav found")
                print("  - Download tested")
                print("  - Upload tested")
            elif result.get("empty"):
                print("\n⚠️  Bucket is empty")
                print("Please upload multi1.png and 1.wav to the bucket")
            else:
                print(f"\n⚠️  Missing files: {result.get('missing_files', [])}")
        else:
            print("❌ S3 ACCESS FAILED!")
            print(f"Error: {result.get('error', 'Unknown error')}")
        print("="*60)