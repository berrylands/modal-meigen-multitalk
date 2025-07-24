"""
Simplified Modal Image that builds successfully
"""

import modal

# Start with the minimal PyTorch image we know works
multitalk_image = (
    modal.Image.debian_slim(python_version="3.10")
    # System packages
    .apt_install([
        "git",
        "ffmpeg", 
        "build-essential",
    ])
    # PyTorch - we know this works
    .pip_install(
        "torch==2.4.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    # Core packages one by one
    .pip_install("transformers==4.49.0")
    .pip_install("huggingface_hub")
    .pip_install("numpy==1.26.4")
    .pip_install("tqdm")
    .pip_install("boto3")  # For AWS
    # Clone MultiTalk repo
    .run_commands(
        "cd /root && git clone https://github.com/MeiGen-AI/MultiTalk.git",
    )
    .env({
        "PYTHONPATH": "/root/MultiTalk:$PYTHONPATH",
    })
)

if __name__ == "__main__":
    print("Simplified Modal image created")
    print("This is a minimal working configuration")