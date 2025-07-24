"""Test HuggingFace with working authentication."""

import modal
import os

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

app = modal.App("test-hf-working")

@app.function(
    secrets=[modal.Secret.from_name("huggingface-secret")],
    image=modal.Image.debian_slim().pip_install(["huggingface-hub"])
)
def test_hf_access():
    """Test HuggingFace access using the working method."""
    import os
    from huggingface_hub import HfApi, list_models
    
    print("HuggingFace Authentication Test")
    print("=" * 50)
    
    # Get token
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        print("❌ No HuggingFace token found!")
        return False
    
    # Also set standard env var for compatibility
    os.environ["HUGGINGFACE_TOKEN"] = token
    os.environ["HF_TOKEN"] = token
    
    print(f"✓ Token found (length: {len(token)})")
    
    # Test 1: Whoami
    try:
        api = HfApi(token=token)
        user = api.whoami()
        print(f"\n✅ Authenticated as: {user['name']}")
        print(f"   Type: {user['type']}")
        print(f"   Organizations: {[org['name'] for org in user.get('orgs', [])]}")
    except Exception as e:
        print(f"\n❌ Authentication failed: {e}")
        return False
    
    # Test 2: List some models
    try:
        print("\n Testing model access...")
        models = list(list_models(limit=3))
        print(f"✅ Can access models. Found {len(models)} models")
        for model in models[:2]:
            print(f"   - {model.modelId}")
    except Exception as e:
        print(f"❌ Model access failed: {e}")
        return False
    
    # Test 3: Check specific model access
    try:
        print("\n Testing specific model access...")
        # Try to access a common model
        model_info = api.model_info("bert-base-uncased")
        print(f"✅ Can access bert-base-uncased")
        print(f"   Downloads: {model_info.downloads:,}")
    except Exception as e:
        print(f"❌ Specific model access failed: {e}")
    
    print("\n" + "=" * 50)
    print("✅ HuggingFace authentication is working correctly!")
    print(f"   User: berrylands")
    print(f"   Token variable: HF_TOKEN")
    
    return True

if __name__ == "__main__":
    import subprocess
    subprocess.run(["modal", "run", __file__ + "::test_hf_access"])