"""Simple Modal test that works with API tokens."""

import os
from dotenv import load_dotenv

# Load environment variables BEFORE importing modal
load_dotenv()

# Set Modal auth token from .env
if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]
    print(f"✓ Modal token loaded from .env")

# Now import modal
import modal

app = modal.App("test-simple")

@app.function()
def hello():
    """Test basic Modal function."""
    return "Hello from Modal!"

# Use a different approach for testing
if __name__ == "__main__":
    print("Testing Modal connection...")
    
    # Use modal run command instead of local_entrypoint
    with app.run():
        result = hello.remote()
        print(f"✅ Success: {result}")
        print("\nModal is properly configured!")