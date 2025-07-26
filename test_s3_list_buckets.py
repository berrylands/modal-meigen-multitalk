"""
List available S3 buckets to find the correct one.
"""

import modal
import os

modal.enable_output()

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

test_image = modal.Image.debian_slim(python_version="3.10").pip_install("boto3")
app = modal.App("test-s3-buckets")

@app.function(
    image=test_image,
    secrets=[modal.Secret.from_name("aws-secret")],
)
def list_buckets():
    """List all available S3 buckets."""
    import boto3
    
    print("Listing available S3 buckets...")
    
    s3 = boto3.client('s3')
    
    try:
        response = s3.list_buckets()
        
        buckets = response['Buckets']
        print(f"\nFound {len(buckets)} buckets:")
        
        for bucket in buckets:
            name = bucket['Name']
            created = bucket['CreationDate'].strftime('%Y-%m-%d')
            print(f"  - {name} (created: {created})")
            
            # Check if this looks like our target bucket
            if 'modal' in name.lower() or 'meigen' in name.lower() or 'multitalk' in name.lower():
                print(f"    ^ This might be the target bucket!")
                
                # Try to list its contents
                try:
                    obj_response = s3.list_objects_v2(Bucket=name, MaxKeys=5)
                    if 'Contents' in obj_response:
                        print(f"    First few objects:")
                        for obj in obj_response['Contents'][:5]:
                            print(f"      - {obj['Key']}")
                except:
                    print(f"    (Unable to list contents)")
        
        return {"success": True, "count": len(buckets)}
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    with app.run():
        result = list_buckets.remote()
        
        if result["success"]:
            print(f"\n✅ Successfully listed {result['count']} buckets")
            print("\nPlease check the bucket names above and use the correct one.")
        else:
            print(f"\n❌ Failed: {result.get('error')}")