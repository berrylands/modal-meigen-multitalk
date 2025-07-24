"""Test Modal with API token."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Debug token
if "MODAL_API_TOKEN" in os.environ:
    print(f"Token found (length: {len(os.environ['MODAL_API_TOKEN'])})")
    print(f"Token format check: contains ':' = {':' in os.environ['MODAL_API_TOKEN']}")
else:
    print("No MODAL_API_TOKEN found in environment")

# Try importing modal after setting token
import modal

try:
    # Create app
    app = modal.App("test-token")
    
    @app.function()
    def test():
        return "Success!"
    
    # Try to run
    with app.run():
        result = test.remote()
        print(f"✅ Result: {result}")
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTrying alternative authentication method...")
    
    # Try setting as environment variable before import
    if "MODAL_API_TOKEN" in os.environ:
        # Try different token formats
        token = os.environ["MODAL_API_TOKEN"]
        
        # Check if we need to create a config file
        config_dir = os.path.expanduser("~/.modal")
        os.makedirs(config_dir, exist_ok=True)
        
        print("Note: Modal API tokens should be set before running the script:")
        print("  export MODAL_AUTH_TOKEN=your_token")
        print("  python test_simple.py")