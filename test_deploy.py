"""Test Modal deployment."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set Modal auth token
if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

import modal

app = modal.App("meigen-multitalk-test")

@app.function()
def health_check():
    """Simple health check."""
    return {"status": "healthy", "message": "Modal deployment successful!"}

@app.function(
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("aws-secret")
    ]
)
def check_secrets():
    """Check if secrets are accessible."""
    import os
    return {
        "huggingface": "HUGGINGFACE_TOKEN" in os.environ,
        "aws_key": "AWS_ACCESS_KEY_ID" in os.environ,
        "aws_secret": "AWS_SECRET_ACCESS_KEY" in os.environ,
        "aws_region": os.environ.get("AWS_REGION", "not set")
    }

if __name__ == "__main__":
    print("Deploying test functions to Modal...")
    print("This will create a persistent deployment.")
    print("Run 'modal app stop meigen-multitalk-test' to stop it later.")