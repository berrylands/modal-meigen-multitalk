"""
Test MultiTalk imports and dependencies within Modal.
"""

import modal

app = modal.App("test-multitalk-imports")

# Use the same image from main app
from modal_meigen_multitalk import multitalk_image

@app.function(image=multitalk_image, gpu="t4")
def test_multitalk_imports():
    """Test if we can import MultiTalk modules."""
    import sys
    sys.path.insert(0, "/root/MultiTalk")
    
    print("Testing MultiTalk imports...")
    print("-" * 60)
    
    # Test basic imports
    try:
        import os
        print(f"✅ MultiTalk directory exists: {os.path.exists('/root/MultiTalk')}")
        
        # List key files
        if os.path.exists('/root/MultiTalk'):
            files = os.listdir('/root/MultiTalk')
            print(f"✅ Found {len(files)} files in MultiTalk directory")
            
            # Check for key modules
            key_modules = ['kokoro', 'models', 'generate_multitalk.py']
            for module in key_modules:
                path = f'/root/MultiTalk/{module}'
                exists = os.path.exists(path)
                print(f"  {module}: {'✅' if exists else '❌'}")
    except Exception as e:
        print(f"❌ Basic checks failed: {e}")
        return {"status": "failed", "error": str(e)}
    
    # Test imports that were failing
    try:
        print("\nTesting problematic imports...")
        
        # Test misaki
        import misaki
        print(f"✅ misaki imported successfully (version: {getattr(misaki, '__version__', 'unknown')})")
        
        # Test misaki submodules
        from misaki import en, espeak
        print("✅ misaki.en and misaki.espeak imported successfully")
        
        # Test kokoro
        from kokoro import KPipeline
        print("✅ kokoro.KPipeline imported successfully")
        
        # Test other MultiTalk imports
        from models.i2v_module import load_models
        print("✅ models.i2v_module imported successfully")
        
        return {"status": "success", "message": "All imports successful"}
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        
        # Try to diagnose the issue
        import subprocess
        print("\nChecking installed packages...")
        result = subprocess.run(["pip", "list", "|", "grep", "-E", "(misaki|kokoro)"], 
                              shell=True, capture_output=True, text=True)
        print(f"Installed packages:\n{result.stdout}")
        
        return {"status": "failed", "error": str(e)}
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return {"status": "failed", "error": str(e)}

@app.local_entrypoint()
def main():
    """Run the import test."""
    print("Running MultiTalk import test...")
    result = test_multitalk_imports.remote()
    print(f"\nTest result: {result}")
    
    if result["status"] == "success":
        print("\n✅ All imports working! Ready to test inference.")
    else:
        print(f"\n❌ Import test failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()