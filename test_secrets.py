"""Test Modal secrets access."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set Modal auth token
if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

import modal

app = modal.App("test-secrets")

@app.function(
    secrets=[modal.Secret.from_name("aws-secret")]
)
def test_aws_secret():
    """Test AWS secret access."""
    import os
    return {
        "aws_access_key_exists": "AWS_ACCESS_KEY_ID" in os.environ,
        "aws_secret_key_exists": "AWS_SECRET_ACCESS_KEY" in os.environ,
        "aws_region": os.environ.get("AWS_REGION", "not set"),
        "key_prefix": os.environ.get("AWS_ACCESS_KEY_ID", "")[:4] + "..." if "AWS_ACCESS_KEY_ID" in os.environ else "missing"
    }

@app.function(
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def test_hf_secret():
    """Test HuggingFace secret access."""
    import os
    return {
        "huggingface_token_exists": "HUGGINGFACE_TOKEN" in os.environ,
        "token_prefix": os.environ.get("HUGGINGFACE_TOKEN", "")[:4] + "..." if "HUGGINGFACE_TOKEN" in os.environ else "missing"
    }

@app.function()
def test_all_secrets():
    """Test all secrets without failing if missing."""
    import os
    results = {}
    
    # Test without loading secrets to check what's available
    print("Checking Modal secrets...")
    
    return {
        "status": "Secrets should be tested individually",
        "note": "Use modal secret list to see available secrets"
    }

if __name__ == "__main__":
    with app.run():
        print("Testing Modal secrets access...\n")
        
        # Test AWS secret
        print("1. AWS Secret:")
        try:
            aws_result = test_aws_secret.remote()
            for key, value in aws_result.items():
                print(f"   ✅ {key}: {value}")
        except modal.exception.NotFoundError:
            print("   ⚠️  AWS secret not found")
            print("   Create it at: https://modal.com/secrets")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Test HuggingFace secret
        print("\n2. HuggingFace Secret:")
        try:
            hf_result = test_hf_secret.remote()
            for key, value in hf_result.items():
                print(f"   ✅ {key}: {value}")
        except modal.exception.NotFoundError:
            print("   ⚠️  HuggingFace secret not found")
            print("   Create it at: https://modal.com/secrets")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print("\n✅ Secret test complete!")