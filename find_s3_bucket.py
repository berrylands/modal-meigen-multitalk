"""
Find S3 bucket with multi1.png and 1.wav files.
"""

import modal
import os

modal.enable_output()

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

test_image = modal.Image.debian_slim(python_version="3.10").pip_install("boto3")
app = modal.App("find-s3-bucket")

@app.function(
    image=test_image,
    secrets=[modal.Secret.from_name("aws-secret")],
    timeout=300,
)
def find_bucket_with_files():
    """Find bucket containing multi1.png and 1.wav."""
    import boto3
    
    print("Searching for bucket with multi1.png and 1.wav...")
    
    s3 = boto3.client('s3')
    
    try:
        response = s3.list_buckets()
        buckets = response['Buckets']
        
        print(f"Checking {len(buckets)} buckets...")
        
        found_buckets = []
        
        for i, bucket in enumerate(buckets):
            name = bucket['Name']
            
            # Show progress every 50 buckets
            if i % 50 == 0:
                print(f"  Checked {i}/{len(buckets)} buckets...")
            
            try:
                # Check if bucket has our files
                obj_response = s3.list_objects_v2(Bucket=name, MaxKeys=100)
                
                if 'Contents' in obj_response:
                    keys = [obj['Key'] for obj in obj_response['Contents']]
                    
                    has_image = 'multi1.png' in keys
                    has_audio = '1.wav' in keys
                    
                    if has_image or has_audio:
                        found_buckets.append({
                            'name': name,
                            'has_image': has_image,
                            'has_audio': has_audio,
                            'total_objects': len(keys),
                            'sample_objects': keys[:5]
                        })
                        
                        print(f"\n✅ Found candidate bucket: {name}")
                        print(f"   multi1.png: {'✅' if has_image else '❌'}")
                        print(f"   1.wav: {'✅' if has_audio else '❌'}")
                        print(f"   Total objects: {len(keys)}")
                        
            except Exception as e:
                # Skip buckets we can't access
                pass
        
        print(f"\n\nSearch complete. Found {len(found_buckets)} candidate buckets.")
        
        return {
            "success": True,
            "found_buckets": found_buckets,
            "total_checked": len(buckets)
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    with app.run():
        print("Starting bucket search...\n")
        result = find_bucket_with_files.remote()
        
        if result["success"]:
            if result["found_buckets"]:
                print("\n" + "="*60)
                print("BUCKETS WITH TARGET FILES:")
                print("="*60)
                
                for bucket in result["found_buckets"]:
                    print(f"\nBucket: {bucket['name']}")
                    print(f"  multi1.png: {'✅' if bucket['has_image'] else '❌'}")
                    print(f"  1.wav: {'✅' if bucket['has_audio'] else '❌'}")
                    print(f"  Total objects: {bucket['total_objects']}")
                    
                    if bucket['has_image'] and bucket['has_audio']:
                        print(f"\n🎯 USE THIS BUCKET: {bucket['name']}")
            else:
                print("\n❌ No buckets found with multi1.png or 1.wav")
        else:
            print(f"\n❌ Search failed: {result.get('error')}")