"""
Test Modal image build with full debug logging enabled.
"""

import modal
import os
import sys

# Maximum debugging setup
sys.stdout.reconfigure(line_buffering=True)
modal.enable_output()
os.environ["MODAL_LOGLEVEL"] = "DEBUG"
os.environ["MODAL_TRACEBACK"] = "1"

# Set up Modal authentication
if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

print("Debug logging enabled:")
print(f"- MODAL_LOGLEVEL: {os.environ.get('MODAL_LOGLEVEL')}")
print(f"- MODAL_TRACEBACK: {os.environ.get('MODAL_TRACEBACK')}")
print(f"- Output enabled: True")
print("-" * 60)

# Use our simplified image
from modal_image_simplified import multitalk_image

app = modal.App("test-debug-logging")

@app.function(
    image=multitalk_image,
    gpu="t4",
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_dict({"MODAL_LOGLEVEL": "DEBUG"})
    ],
)
def test_environment():
    """Test environment with debug logging."""
    import torch
    import transformers
    import os
    
    print("=== Remote Function Execution ===")
    print(f"Debug level in container: {os.environ.get('MODAL_LOGLEVEL', 'Not set')}")
    
    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "transformers_version": transformers.__version__,
        "hf_token_available": any(k in os.environ for k in ["HF_TOKEN", "HUGGINGFACE_TOKEN"]),
        "multitalk_exists": os.path.exists("/root/MultiTalk"),
    }

if __name__ == "__main__":
    print("\nStarting Modal app with debug logging...")
    
    try:
        with modal.enable_output():
            with app.run():
                print("\nApp started, running test function...")
                result = test_environment.remote()
                print(f"\nTest completed successfully!")
                print(f"Result: {result}")
    except Exception as e:
        print(f"\n❌ Error occurred: {type(e).__name__}")
        print(f"Message: {str(e)}")
        
        # Try to get more information
        if hasattr(e, '__cause__'):
            print(f"Cause: {e.__cause__}")
        if hasattr(e, '__context__'):
            print(f"Context: {e.__context__}")
        
        print("\nFull traceback enabled - check above for details")
        
        # Additional debugging info
        print("\nDebugging suggestions:")
        print("1. Check the Modal dashboard at https://modal.com/")
        print("2. Look for the app in your workspace")
        print("3. Click on the failed build to see detailed logs")
        print("4. Try running: modal app logs test-debug-logging")
        
        raise  # Re-raise to see full traceback