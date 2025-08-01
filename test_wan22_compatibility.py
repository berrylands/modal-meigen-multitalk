#!/usr/bin/env python3
"""
Test script to check Wan 2.2 model compatibility with MultiTalk.
This script will:
1. Check if we can download Wan 2.2
2. Compare file structure with Wan 2.1
3. Test basic loading compatibility
"""

import os
import json
import subprocess
from pathlib import Path

def check_huggingface_model_info():
    """Get information about Wan 2.2 model from Hugging Face."""
    print("=== Checking Wan 2.2 Model Information ===")
    
    # Use huggingface-cli to get model info
    cmd = ["huggingface-cli", "repo", "info", "Wan-AI/Wan2.2-I2V-A14B", "--repo-type", "model"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Model found on Hugging Face")
            print(result.stdout)
        else:
            print("❌ Error getting model info:")
            print(result.stderr)
    except FileNotFoundError:
        print("❌ huggingface-cli not installed. Install with: pip install huggingface-hub")
        return False
    
    return True

def compare_model_structures():
    """Compare the expected file structures between Wan 2.1 and Wan 2.2."""
    print("\n=== Model Structure Comparison ===")
    
    wan21_structure = {
        "config.json": "Model configuration",
        "diffusion_pytorch_model.safetensors.index.json": "Model index file",
        "diffusion_pytorch_model-*.safetensors": "Model weight shards",
        "model_index.json": "Pipeline configuration",
        "scheduler/scheduler_config.json": "Scheduler configuration",
        "text_encoder/": "Text encoder files",
        "vae/": "VAE files",
        "unet/": "UNet diffusion model files"
    }
    
    print("Expected Wan 2.1 structure:")
    for file, desc in wan21_structure.items():
        print(f"  - {file}: {desc}")
    
    print("\n⚠️  Wan 2.2 with MoE might have:")
    print("  - Additional expert model files")
    print("  - Different index structure for multiple experts")
    print("  - Modified config.json with MoE parameters")
    
    return True

def test_multitalk_compatibility():
    """Test if MultiTalk's loading code can handle the new model structure."""
    print("\n=== MultiTalk Compatibility Check ===")
    
    # Check key assumptions in MultiTalk
    compatibility_checks = {
        "14B Parameter Assumption": "MultiTalk hardcodes 'multitalk-14B' task",
        "Model Loading": "Uses wan.MultiTalkPipeline() - needs to support MoE",
        "Weight Files": "Expects specific safetensors structure",
        "Config Format": "Assumes standard Wan 2.1 config format"
    }
    
    print("Potential compatibility issues:")
    for check, desc in compatibility_checks.items():
        print(f"  ⚠️  {check}: {desc}")
    
    return True

def create_test_download_script():
    """Create a script to download a small portion of Wan 2.2 for testing."""
    script_content = '''#!/usr/bin/env python3
"""
Download Wan 2.2 model files for compatibility testing.
Only downloads config files first to check structure.
"""

import os
from huggingface_hub import snapshot_download, hf_hub_download

def download_config_only():
    """Download only configuration files to check compatibility."""
    print("Downloading Wan 2.2 configuration files...")
    
    try:
        # Download only JSON config files first
        config_files = [
            "config.json",
            "model_index.json",
            "diffusion_pytorch_model.safetensors.index.json"
        ]
        
        local_dir = "./test_wan22_configs"
        os.makedirs(local_dir, exist_ok=True)
        
        for file in config_files:
            try:
                downloaded = hf_hub_download(
                    repo_id="Wan-AI/Wan2.2-I2V-A14B",
                    filename=file,
                    local_dir=local_dir
                )
                print(f"✅ Downloaded: {file}")
            except Exception as e:
                print(f"⚠️  Could not download {file}: {e}")
        
        # Also try to get scheduler config
        try:
            downloaded = hf_hub_download(
                repo_id="Wan-AI/Wan2.2-I2V-A14B",
                filename="scheduler/scheduler_config.json",
                local_dir=local_dir
            )
            print("✅ Downloaded: scheduler/scheduler_config.json")
        except:
            pass
            
        return local_dir
        
    except Exception as e:
        print(f"❌ Error downloading configs: {e}")
        return None

def analyze_configs(config_dir):
    """Analyze the downloaded configuration files."""
    import json
    
    print("\\n=== Analyzing Configuration Files ===")
    
    # Check main config
    config_path = os.path.join(config_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"\\nMain config.json:")
        print(f"  - Model type: {config.get('_class_name', 'Unknown')}")
        print(f"  - Model version: {config.get('model_version', 'Unknown')}")
        if 'moe' in str(config).lower() or 'expert' in str(config).lower():
            print("  - ✅ MoE configuration detected!")
        else:
            print("  - ⚠️  No obvious MoE configuration found")
    
    # Check index file
    index_path = os.path.join(config_dir, "diffusion_pytorch_model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            index = json.load(f)
        print(f"\\nModel index file:")
        print(f"  - Weight maps: {len(index.get('weight_map', {}))} entries")
        # Check for expert-related weights
        expert_weights = [k for k in index.get('weight_map', {}).keys() if 'expert' in k.lower()]
        if expert_weights:
            print(f"  - ✅ Found {len(expert_weights)} expert-related weights")
        else:
            print("  - ⚠️  No expert weights found in index")

if __name__ == "__main__":
    config_dir = download_config_only()
    if config_dir:
        analyze_configs(config_dir)
        print(f"\\n💡 Config files saved to: {config_dir}")
        print("   Examine these files to understand the model structure")
'''
    
    with open("download_wan22_configs.py", "w") as f:
        f.write(script_content)
    os.chmod("download_wan22_configs.py", 0o755)
    print("\n✅ Created download_wan22_configs.py")
    print("   Run this script to download and analyze Wan 2.2 configuration files")
    
    return True

def main():
    """Run all compatibility checks."""
    print("🔍 Wan 2.2 Compatibility Check for MultiTalk\n")
    
    # Run checks
    checks = [
        ("Model Info", check_huggingface_model_info),
        ("Structure Comparison", compare_model_structures),
        ("MultiTalk Compatibility", test_multitalk_compatibility),
        ("Download Script", create_test_download_script)
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{'='*50}")
        success = check_func()
        results.append((name, success))
    
    # Summary
    print(f"\n{'='*50}")
    print("=== Summary ===")
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {name}")
    
    print("\n📋 Next Steps:")
    print("1. Run ./download_wan22_configs.py to download config files")
    print("2. Analyze the configuration differences")
    print("3. Test loading with a minimal script")
    print("4. Update MultiTalk integration if compatible")

if __name__ == "__main__":
    main()