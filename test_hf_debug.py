"""Debug HuggingFace secret."""

import modal
import os

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

app = modal.App("test-hf-debug")

@app.function(
    secrets=[modal.Secret.from_name("huggingface-secret")],
    image=modal.Image.debian_slim().pip_install("requests")
)
def debug_hf_secret():
    """Debug what environment variables are available."""
    import os
    
    print("HuggingFace Secret Debug")
    print("=" * 50)
    
    # List all environment variables that might contain HF token
    print("\nEnvironment variables containing 'HUG' or 'HF':")
    for key in sorted(os.environ.keys()):
        if any(x in key.upper() for x in ["HUG", "HF", "TOKEN"]):
            # Mask the value for security
            value = os.environ[key]
            if len(value) > 10:
                masked = value[:4] + "..." + value[-4:]
            else:
                masked = "***"
            print(f"  {key}: {masked}")
    
    # Common HF token variable names
    common_names = [
        "HUGGINGFACE_TOKEN",
        "HUGGING_FACE_TOKEN", 
        "HF_TOKEN",
        "HUGGINGFACE_API_TOKEN",
        "HF_API_TOKEN"
    ]
    
    print("\nChecking common HF token names:")
    for name in common_names:
        exists = name in os.environ
        print(f"  {name}: {'✅ exists' if exists else '❌ not found'}")
    
    # Check if we can access HF API with any token found
    for name in common_names:
        if name in os.environ:
            print(f"\nTesting API with {name}...")
            try:
                import requests
                headers = {"Authorization": f"Bearer {os.environ[name]}"}
                response = requests.get(
                    "https://huggingface.co/api/whoami",
                    headers=headers,
                    timeout=10
                )
                if response.status_code == 200:
                    print(f"  ✅ Success! User: {response.json().get('name', 'unknown')}")
                else:
                    print(f"  ❌ Failed with status: {response.status_code}")
            except Exception as e:
                print(f"  ❌ Error: {e}")
            break

if __name__ == "__main__":
    import subprocess
    subprocess.run(["modal", "run", __file__ + "::debug_hf_secret"])