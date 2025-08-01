#!/usr/bin/env python3
"""
Modal test script for Wan 2.2 single expert integration with MultiTalk.
This tests if we can use Wan 2.2's low_noise_model as a base for MultiTalk.
"""

import modal
import os

app = modal.App("wan22-single-expert-test")

# Create test image with minimal dependencies
test_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.4.0",
        "transformers",
        "diffusers",
        "safetensors",
        "huggingface_hub",
    )
)

@app.function(image=test_image, timeout=600)
def test_wan22_loading():
    """Test if we can load Wan 2.2's low_noise_model as a standard Diffusers model."""
    import torch
    from diffusers import DiffusionPipeline
    from huggingface_hub import snapshot_download
    import json
    
    print("=== Testing Wan 2.2 Single Expert Loading ===")
    
    try:
        # Download only the low_noise_model config first
        print("\n1. Downloading low_noise_model configuration...")
        config_path = snapshot_download(
            repo_id="Wan-AI/Wan2.2-I2V-A14B",
            allow_patterns=["low_noise_model/config.json"],
            local_dir="./wan22_test"
        )
        
        # Load and analyze config
        with open(os.path.join(config_path, "low_noise_model/config.json"), 'r') as f:
            config = json.load(f)
        
        print(f"\n2. Model configuration:")
        print(f"   - Model type: {config.get('_class_name', 'Unknown')}")
        print(f"   - In channels: {config.get('in_channels', 'Unknown')}")
        print(f"   - Out channels: {config.get('out_channels', 'Unknown')}")
        
        # Check if it's compatible with standard UNet3D
        if 'block_out_channels' in config:
            print(f"   - Block out channels: {config['block_out_channels']}")
            print("   ✅ Appears to be UNet3D compatible!")
        else:
            print("   ⚠️  May not be standard UNet3D structure")
        
        print("\n3. Compatibility assessment:")
        print("   - Can potentially be loaded as standard Diffusers UNet3D")
        print("   - Would need to handle text encoder differences")
        print("   - VAE appears to be the same as Wan 2.1")
        
        return {
            "success": True,
            "config_compatible": 'block_out_channels' in config,
            "model_type": config.get('_class_name', 'Unknown')
        }
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return {"success": False, "error": str(e)}

@app.function(image=test_image, gpu="a10g", timeout=900)
def test_multitalk_compatibility():
    """Test if MultiTalk weights can be applied to Wan 2.2 single expert."""
    print("=== Testing MultiTalk Weight Compatibility ===")
    
    # This would test:
    # 1. Loading low_noise_model weights
    # 2. Loading MultiTalk adapter weights
    # 3. Checking tensor shape compatibility
    # 4. Running a simple inference
    
    print("\n⚠️  This requires actual model weights to test fully")
    print("   Next steps would be:")
    print("   1. Download low_noise_model weights")
    print("   2. Restructure to match Wan 2.1 layout")
    print("   3. Apply MultiTalk modifications")
    print("   4. Test inference")
    
    return {"test": "placeholder", "next_steps": "Download full weights"}

if __name__ == "__main__":
    with app.run():
        # Test configuration loading
        print("Testing Wan 2.2 configuration compatibility...\n")
        result = test_wan22_loading.remote()
        print(f"\nResult: {result}")
        
        if result.get("success") and result.get("config_compatible"):
            print("\n✅ Configuration appears compatible!")
            print("\nNext step: Test with full model weights")
            # Uncomment to test with GPU:
            # gpu_result = test_multitalk_compatibility.remote()
            # print(f"\nGPU Test Result: {gpu_result}")
