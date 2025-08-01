#!/usr/bin/env python3
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
    
    print("\n=== Analyzing Configuration Files ===")
    
    # Check main config
    config_path = os.path.join(config_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"\nMain config.json:")
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
        print(f"\nModel index file:")
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
        print(f"\n💡 Config files saved to: {config_dir}")
        print("   Examine these files to understand the model structure")
