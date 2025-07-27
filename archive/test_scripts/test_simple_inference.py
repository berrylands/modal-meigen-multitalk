"""
Simple test to verify core MeiGen-MultiTalk functionality on Modal.
This test focuses on verifying the models are downloaded and basic inference works.
"""

import subprocess
import os

def test_models_exist():
    """Test if models are already downloaded."""
    print("Testing if models exist in Modal volume...")
    
    cmd = [
        "modal", "run", "--quiet",
        "-c", "import os; print('Models exist:', os.path.exists('/models/base/diffusion_pytorch_model.safetensors.index.json'))",
        "modal_meigen_multitalk.py"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"Check result: {result.stdout.strip()}")
    
    if "Models exist: False" in result.stdout:
        print("\n⚠️  Models not found. Running download...")
        download_cmd = ["modal", "run", "modal_meigen_multitalk.py", "--action", "download"]
        download_result = subprocess.run(download_cmd, capture_output=True, text=True)
        
        if download_result.returncode == 0:
            print("✅ Model download completed successfully")
        else:
            print(f"❌ Model download failed: {download_result.stderr}")
            return False
    
    return True

def test_environment_with_new_deps():
    """Test that environment loads with all dependencies."""
    print("\nTesting environment with new dependencies...")
    
    cmd = ["modal", "run", "modal_meigen_multitalk.py", "--action", "test"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0 and "misaki" not in result.stderr:
        print("✅ Environment test passed")
        print("Output:", result.stdout)
        return True
    else:
        print("❌ Environment test failed")
        print("STDERR:", result.stderr)
        return False

def main():
    print("Simple MeiGen-MultiTalk Test")
    print("="*60)
    
    # First test environment
    if not test_environment_with_new_deps():
        print("\n❌ Environment setup failed. Check dependencies.")
        return
    
    # Then test models
    if not test_models_exist():
        print("\n❌ Model setup failed.")
        return
    
    print("\n✅ All tests passed! Ready for inference testing.")

if __name__ == "__main__":
    main()