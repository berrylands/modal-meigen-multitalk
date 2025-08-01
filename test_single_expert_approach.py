#!/usr/bin/env python3
"""
Test the Single Expert Approach for Wan 2.2 integration.
This script will attempt to use the low_noise_model as a drop-in replacement for Wan 2.1.
"""

import os
import shutil
from pathlib import Path

def create_wan22_single_expert_structure():
    """
    Create a directory structure that mimics Wan 2.1 using Wan 2.2's low_noise_model.
    This is a proof of concept for the single expert approach.
    """
    print("=== Creating Wan 2.2 Single Expert Structure ===\n")
    
    # Define paths
    wan22_base = "./wan22_test"
    single_expert_dir = "./wan22_single_expert"
    
    # Create directory
    os.makedirs(single_expert_dir, exist_ok=True)
    
    print("📁 Directory structure for single expert approach:")
    print(f"   Source: {wan22_base}/low_noise_model/")
    print(f"   Target: {single_expert_dir}/")
    
    # Files to map from Wan 2.2 structure to Wan 2.1 structure
    file_mappings = {
        # Wan 2.2 low_noise_model -> Wan 2.1 compatible structure
        "low_noise_model/config.json": "config.json",
        "low_noise_model/diffusion_pytorch_model.safetensors.index.json": "diffusion_pytorch_model.safetensors.index.json",
        "low_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors": "diffusion_pytorch_model-00001-of-00006.safetensors",
        # Add other weight files as needed
        "Wan2.1_VAE.pth": "vae/diffusion_pytorch_model.safetensors",  # VAE is shared
        "google/umt5-xxl/": "text_encoder/",  # Text encoder mapping
    }
    
    print("\n📋 File mapping plan:")
    for src, dst in file_mappings.items():
        print(f"   {src} → {dst}")
    
    return single_expert_dir, file_mappings

def create_integration_test_script():
    """Create a test script for Modal to test Wan 2.2 single expert integration."""
    
    script_content = '''#!/usr/bin/env python3
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
        print("\\n1. Downloading low_noise_model configuration...")
        config_path = snapshot_download(
            repo_id="Wan-AI/Wan2.2-I2V-A14B",
            allow_patterns=["low_noise_model/config.json"],
            local_dir="./wan22_test"
        )
        
        # Load and analyze config
        with open(os.path.join(config_path, "low_noise_model/config.json"), 'r') as f:
            config = json.load(f)
        
        print(f"\\n2. Model configuration:")
        print(f"   - Model type: {config.get('_class_name', 'Unknown')}")
        print(f"   - In channels: {config.get('in_channels', 'Unknown')}")
        print(f"   - Out channels: {config.get('out_channels', 'Unknown')}")
        
        # Check if it's compatible with standard UNet3D
        if 'block_out_channels' in config:
            print(f"   - Block out channels: {config['block_out_channels']}")
            print("   ✅ Appears to be UNet3D compatible!")
        else:
            print("   ⚠️  May not be standard UNet3D structure")
        
        print("\\n3. Compatibility assessment:")
        print("   - Can potentially be loaded as standard Diffusers UNet3D")
        print("   - Would need to handle text encoder differences")
        print("   - VAE appears to be the same as Wan 2.1")
        
        return {
            "success": True,
            "config_compatible": 'block_out_channels' in config,
            "model_type": config.get('_class_name', 'Unknown')
        }
        
    except Exception as e:
        print(f"\\n❌ Error: {e}")
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
    
    print("\\n⚠️  This requires actual model weights to test fully")
    print("   Next steps would be:")
    print("   1. Download low_noise_model weights")
    print("   2. Restructure to match Wan 2.1 layout")
    print("   3. Apply MultiTalk modifications")
    print("   4. Test inference")
    
    return {"test": "placeholder", "next_steps": "Download full weights"}

if __name__ == "__main__":
    with app.run():
        # Test configuration loading
        print("Testing Wan 2.2 configuration compatibility...\\n")
        result = test_wan22_loading.remote()
        print(f"\\nResult: {result}")
        
        if result.get("success") and result.get("config_compatible"):
            print("\\n✅ Configuration appears compatible!")
            print("\\nNext step: Test with full model weights")
            # Uncomment to test with GPU:
            # gpu_result = test_multitalk_compatibility.remote()
            # print(f"\\nGPU Test Result: {gpu_result}")
'''
    
    with open("test_wan22_modal.py", "w") as f:
        f.write(script_content)
    os.chmod("test_wan22_modal.py", 0o755)
    
    print("\n✅ Created test_wan22_modal.py")
    print("   This script can be run on Modal to test compatibility")
    
    return "test_wan22_modal.py"

def create_local_test():
    """Create a local test to check config compatibility."""
    
    script_content = '''#!/usr/bin/env python3
"""
Local test to analyze Wan 2.2 low_noise_model configuration.
"""

import json
import os
from huggingface_hub import hf_hub_download

def download_and_analyze_config():
    """Download and analyze the low_noise_model configuration."""
    print("=== Wan 2.2 Low Noise Model Analysis ===\\n")
    
    try:
        # Download config
        print("Downloading low_noise_model config...")
        config_file = hf_hub_download(
            repo_id="Wan-AI/Wan2.2-I2V-A14B",
            filename="low_noise_model/config.json",
            local_dir="./wan22_configs"
        )
        
        # Load config
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Analyze structure
        print("\\n📊 Model Configuration:")
        important_keys = [
            '_class_name', 'act_fn', 'attention_head_dim', 
            'block_out_channels', 'down_block_types', 'in_channels',
            'layers_per_block', 'norm_num_groups', 'out_channels',
            'up_block_types', 'sample_size'
        ]
        
        for key in important_keys:
            if key in config:
                value = config[key]
                if isinstance(value, list) and len(value) > 3:
                    value = f"{value[:3]}... ({len(value)} items)"
                print(f"  {key}: {value}")
        
        # Compare with expected Wan 2.1 structure
        print("\\n🔍 Compatibility Check:")
        
        # Check if it's a UNet3DConditionModel
        if config.get('_class_name') == 'UNet3DConditionModel':
            print("  ✅ Model type matches (UNet3DConditionModel)")
        else:
            print(f"  ⚠️  Model type: {config.get('_class_name')} (expected UNet3DConditionModel)")
        
        # Check dimensions
        if config.get('in_channels') == 10:  # Typical for video models with conditioning
            print("  ✅ Input channels match expected value")
        else:
            print(f"  ⚠️  Input channels: {config.get('in_channels')} (expected 10)")
        
        # Save formatted config for reference
        output_path = "./wan22_configs/low_noise_config_formatted.json"
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\\n💾 Saved formatted config to: {output_path}")
        
        return config
        
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        return None

if __name__ == "__main__":
    config = download_and_analyze_config()
    
    if config:
        print("\\n✅ Analysis complete!")
        print("\\n📋 Next steps:")
        print("1. Compare this config with Wan 2.1 config")
        print("2. Check if MultiTalk modifications are compatible")
        print("3. Test loading with actual weights")
'''
    
    with open("analyze_wan22_config.py", "w") as f:
        f.write(script_content)
    os.chmod("analyze_wan22_config.py", 0o755)
    
    print("\n✅ Created analyze_wan22_config.py")
    return "analyze_wan22_config.py"

def main():
    """Run the single expert approach test setup."""
    print("🧪 Wan 2.2 Single Expert Approach Test Setup\n")
    print("="*60 + "\n")
    
    # Create directory structure plan
    single_expert_dir, file_mappings = create_wan22_single_expert_structure()
    
    # Create test scripts
    print("\n=== Creating Test Scripts ===")
    modal_script = create_integration_test_script()
    local_script = create_local_test()
    
    # Summary
    print("\n" + "="*60)
    print("\n📊 Test Setup Complete!")
    print("\nCreated files:")
    print(f"  1. {modal_script} - Modal test for full compatibility")
    print(f"  2. {local_script} - Local config analysis")
    
    print("\n🚀 To proceed:")
    print("1. Run ./analyze_wan22_config.py to analyze low_noise_model config")
    print("2. Compare with Wan 2.1 structure")
    print("3. If compatible, download full weights and test on Modal")
    print("4. Run ./test_wan22_modal.py to test integration")
    
    print("\n⚡ Quick start:")
    print("   python analyze_wan22_config.py")

if __name__ == "__main__":
    main()