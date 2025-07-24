"""Diagnose HuggingFace token issues."""

import modal
import os

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

app = modal.App("diagnose-hf")

@app.function(
    secrets=[modal.Secret.from_name("huggingface-secret")],
    image=modal.Image.debian_slim().pip_install(["requests", "huggingface-hub"])
)
def diagnose_token():
    """Diagnose HF token issues."""
    import os
    import requests
    from huggingface_hub import HfApi
    
    print("HuggingFace Token Diagnostics")
    print("=" * 60)
    
    # Find the token
    token = None
    token_var = None
    for var in ["HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_TOKEN"]:
        if var in os.environ:
            token = os.environ[var]
            token_var = var
            print(f"\n✓ Found token in: {var}")
            break
    
    if not token:
        print("❌ No HuggingFace token found!")
        return
    
    # Token analysis
    print(f"\nToken Analysis:")
    print(f"  Length: {len(token)} characters")
    print(f"  Starts with 'hf_': {token.startswith('hf_')}")
    print(f"  Contains spaces: {' ' in token}")
    has_newline = '\n' in token
    print(f"  Contains newlines: {has_newline}")
    
    # Clean token (remove any whitespace)
    clean_token = token.strip()
    if clean_token != token:
        print(f"  ⚠️  Token has extra whitespace!")
        token = clean_token
    
    # Test 1: Direct API call
    print(f"\n1. Testing direct API call:")
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(
            "https://huggingface.co/api/whoami",
            headers=headers,
            timeout=10
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Success! User: {data.get('name', 'unknown')}")
            print(f"   Type: {data.get('type', 'unknown')}")
        else:
            print(f"   ❌ Error: {response.text[:200]}")
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    # Test 2: Using huggingface-hub
    print(f"\n2. Testing with huggingface-hub library:")
    try:
        # Set token in environment for library
        os.environ["HUGGINGFACE_TOKEN"] = token
        os.environ["HF_TOKEN"] = token
        
        api = HfApi(token=token)
        user = api.whoami()
        print(f"   ✅ Success! User: {user['name']}")
        print(f"   Organizations: {[org['name'] for org in user.get('orgs', [])]}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Try without Bearer prefix
    print(f"\n3. Testing without Bearer prefix:")
    try:
        headers = {"Authorization": token}  # No "Bearer " prefix
        response = requests.get(
            "https://huggingface.co/api/whoami",
            headers=headers,
            timeout=10
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   ✅ Works without Bearer prefix!")
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    # Recommendations
    print(f"\n" + "=" * 60)
    print("Recommendations:")
    print("1. Make sure the token starts with 'hf_'")
    print("2. Ensure no extra whitespace or newlines")
    print("3. Check token permissions at: https://huggingface.co/settings/tokens")
    print("4. Token should have at least 'read' scope")

if __name__ == "__main__":
    import subprocess
    subprocess.run(["modal", "run", __file__ + "::diagnose_token"])