"""Test AWS secret access only."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set Modal auth token
if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

import modal

app = modal.App("test-aws-only")

@app.function(
    secrets=[modal.Secret.from_name("aws-secret")]
)
def test_aws():
    """Test AWS secret access and S3 connectivity."""
    import os
    import boto3
    
    result = {
        "aws_access_key_exists": "AWS_ACCESS_KEY_ID" in os.environ,
        "aws_secret_key_exists": "AWS_SECRET_ACCESS_KEY" in os.environ,
        "aws_region": os.environ.get("AWS_REGION", "not set"),
    }
    
    # Try to create S3 client
    if result["aws_access_key_exists"] and result["aws_secret_key_exists"]:
        try:
            s3 = boto3.client('s3', region_name=os.environ.get("AWS_REGION", "eu-west-1"))
            # Try to list buckets (requires valid credentials)
            buckets = s3.list_buckets()
            result["s3_access"] = "✅ S3 access working"
            result["bucket_count"] = len(buckets.get('Buckets', []))
        except Exception as e:
            result["s3_access"] = f"❌ S3 error: {str(e)[:50]}..."
    else:
        result["s3_access"] = "❌ Missing credentials"
    
    return result

if __name__ == "__main__":
    with app.run():
        print("Testing AWS secret access...\n")
        
        try:
            result = test_aws.remote()
            for key, value in result.items():
                print(f"• {key}: {value}")
            
            print("\n✅ AWS secret test complete!")
        except Exception as e:
            print(f"❌ Error: {e}")
            print("\nMake sure aws-secret is created in Modal dashboard")