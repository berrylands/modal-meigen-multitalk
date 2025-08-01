#!/usr/bin/env python3
"""
Check what files are actually available in the Wan 2.2 repository.
"""

from huggingface_hub import HfApi, list_repo_files

def list_wan22_files():
    """List all files in the Wan 2.2 repository."""
    api = HfApi()
    
    try:
        print("=== Files in Wan-AI/Wan2.2-I2V-A14B repository ===\n")
        
        # List all files in the repo
        files = list_repo_files("Wan-AI/Wan2.2-I2V-A14B")
        
        # Group files by directory
        root_files = []
        directories = {}
        
        for file in sorted(files):
            if '/' in file:
                dir_name = file.split('/')[0]
                if dir_name not in directories:
                    directories[dir_name] = []
                directories[dir_name].append(file)
            else:
                root_files.append(file)
        
        # Print root files
        if root_files:
            print("Root files:")
            for f in root_files:
                print(f"  - {f}")
            print()
        
        # Print directories and their files
        for dir_name, dir_files in sorted(directories.items()):
            print(f"{dir_name}/:")
            # Show first 5 files in each directory
            for f in dir_files[:5]:
                print(f"  - {f}")
            if len(dir_files) > 5:
                print(f"  ... and {len(dir_files) - 5} more files")
            print()
        
        # Look for config files
        print("\n=== Key Configuration Files ===")
        config_files = [f for f in files if 'config' in f.lower() or 'index' in f.lower()]
        if config_files:
            for f in config_files:
                print(f"  - {f}")
        else:
            print("  ⚠️  No obvious config files found")
            
        # Check for MoE-related files
        print("\n=== MoE/Expert Related Files ===")
        moe_files = [f for f in files if 'expert' in f.lower() or 'moe' in f.lower()]
        if moe_files:
            for f in moe_files[:10]:  # Show first 10
                print(f"  - {f}")
            if len(moe_files) > 10:
                print(f"  ... and {len(moe_files) - 10} more files")
        else:
            print("  ⚠️  No obvious MoE/expert files found")
            
        return files
        
    except Exception as e:
        print(f"❌ Error listing repository files: {e}")
        return None

def download_key_files(files):
    """Download key configuration files based on what's available."""
    from huggingface_hub import hf_hub_download
    import os
    
    print("\n=== Downloading Available Config Files ===")
    
    # Look for any JSON files that might be configs
    json_files = [f for f in files if f.endswith('.json') and '/' not in f]
    
    local_dir = "./test_wan22_configs"
    os.makedirs(local_dir, exist_ok=True)
    
    for file in json_files[:5]:  # Download up to 5 JSON files
        try:
            downloaded = hf_hub_download(
                repo_id="Wan-AI/Wan2.2-I2V-A14B",
                filename=file,
                local_dir=local_dir
            )
            print(f"✅ Downloaded: {file}")
        except Exception as e:
            print(f"❌ Could not download {file}: {e}")
    
    # Try to download README if exists
    if "README.md" in files:
        try:
            downloaded = hf_hub_download(
                repo_id="Wan-AI/Wan2.2-I2V-A14B",
                filename="README.md",
                local_dir=local_dir
            )
            print(f"✅ Downloaded: README.md")
            
            # Read and display key info
            with open(os.path.join(local_dir, "README.md"), 'r') as f:
                content = f.read()
                print("\n=== README Content (first 500 chars) ===")
                print(content[:500])
        except Exception as e:
            print(f"❌ Could not download README.md: {e}")

if __name__ == "__main__":
    files = list_wan22_files()
    if files:
        download_key_files(files)
        print(f"\n📊 Total files in repository: {len(files)}")