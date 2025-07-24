"""Test all Modal secrets."""

import modal
import os

# Set auth token if running locally
if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

app = modal.App("test-all-secrets")

@app.function(
    secrets=[
        modal.Secret.from_name("aws-secret"),
        modal.Secret.from_name("huggingface-secret")
    ]
)
def check_all_secrets():
    """Check all secrets at once."""
    import os
    
    print("Modal Secrets Test")
    print("=" * 50)
    
    # Check AWS
    print("\n1. AWS Secret:")
    print(f"   AWS_ACCESS_KEY_ID exists: {'AWS_ACCESS_KEY_ID' in os.environ}")
    print(f"   AWS_SECRET_ACCESS_KEY exists: {'AWS_SECRET_ACCESS_KEY' in os.environ}")
    print(f"   AWS_REGION: {os.environ.get('AWS_REGION', 'not set')}")
    
    # Check HuggingFace
    print("\n2. HuggingFace Secret:")
    print(f"   HUGGINGFACE_TOKEN exists: {'HUGGINGFACE_TOKEN' in os.environ}")
    if "HUGGINGFACE_TOKEN" in os.environ:
        token = os.environ["HUGGINGFACE_TOKEN"]
        print(f"   Token prefix: {token[:8]}...")
        print(f"   Token length: {len(token)} characters")
    
    # Test HuggingFace API
    if "HUGGINGFACE_TOKEN" in os.environ:
        try:
            import requests
            headers = {"Authorization": f"Bearer {os.environ['HUGGINGFACE_TOKEN']}"}
            response = requests.get(
                "https://huggingface.co/api/whoami", 
                headers=headers,
                timeout=10
            )
            if response.status_code == 200:
                user_info = response.json()
                print(f"   ✅ HF API access verified for user: {user_info.get('name', 'unknown')}")
            else:
                print(f"   ❌ HF API returned status: {response.status_code}")
        except Exception as e:
            print(f"   ❌ HF API test failed: {e}")
    
    return {
        "aws_configured": all(k in os.environ for k in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]),
        "huggingface_configured": "HUGGINGFACE_TOKEN" in os.environ
    }

if __name__ == "__main__":
    # Run with modal CLI
    import subprocess
    subprocess.run(["modal", "run", __file__ + "::check_all_secrets"])