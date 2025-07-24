"""Simple Modal test without secrets."""

import modal
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set Modal token if available
if "MODAL_API_TOKEN" in os.environ:
    # Modal API tokens are in format "id:secret"
    if ":" in os.environ["MODAL_API_TOKEN"]:
        parts = os.environ["MODAL_API_TOKEN"].split(":")
        if len(parts) == 2:
            os.environ["MODAL_TOKEN_ID"] = parts[0]
            os.environ["MODAL_TOKEN_SECRET"] = parts[1]
    else:
        # If it's a single token, use it as the secret
        os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

app = modal.App("test-simple")

@app.function()
def hello():
    """Test basic Modal function."""
    return "Hello from Modal!"

@app.local_entrypoint()
def main():
    """Run the test."""
    print("Testing Modal connection...")
    try:
        result = hello.remote()
        print(f"✅ Success: {result}")
        print("\nModal is properly configured!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nPlease run 'modal setup' to authenticate with Modal")

if __name__ == "__main__":
    main()