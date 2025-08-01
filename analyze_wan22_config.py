#!/usr/bin/env python3
"""
Local test to analyze Wan 2.2 low_noise_model configuration.
"""

import json
import os
from huggingface_hub import hf_hub_download

def download_and_analyze_config():
    """Download and analyze the low_noise_model configuration."""
    print("=== Wan 2.2 Low Noise Model Analysis ===\n")
    
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
        print("\n📊 Model Configuration:")
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
        print("\n🔍 Compatibility Check:")
        
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
        print(f"\n💾 Saved formatted config to: {output_path}")
        
        return config
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None

if __name__ == "__main__":
    config = download_and_analyze_config()
    
    if config:
        print("\n✅ Analysis complete!")
        print("\n📋 Next steps:")
        print("1. Compare this config with Wan 2.1 config")
        print("2. Check if MultiTalk modifications are compatible")
        print("3. Test loading with actual weights")
