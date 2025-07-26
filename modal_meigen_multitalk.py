"""
MeiGen-MultiTalk on Modal - Proper Implementation
Following Modal best practices for ML deployments
"""

import modal
from pathlib import Path

# Create the Modal app
app = modal.App("meigen-multitalk")

# Define volumes for model storage
model_volume = modal.Volume.from_name("multitalk-models", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("multitalk-hf-cache", create_if_missing=True)

# Image definition - matching Colab versions exactly
multitalk_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["git", "ffmpeg"])  # Minimal system deps
    .pip_install(
        # PyTorch with CUDA 12.1
        "torch==2.4.1",
        "torchvision==0.19.1", 
        "torchaudio==2.4.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        # xformers with CUDA 12.1
        "xformers==0.0.28",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        # Other packages from PyPI
        "transformers==4.49.0",
        "huggingface_hub",
        "accelerate",
        "diffusers>=0.30.0",
        "librosa",
        "moviepy",
        "opencv-python",
        "numpy==1.26.4",
        "einops",
        "omegaconf",
        "tqdm",
        "misaki[en]",  # G2P engine for TTS (English support)
        "optimum-quanto==0.2.6",  # Additional MultiTalk deps
        "easydict",
        "ftfy",
        "pyloudnorm",
        "scikit-image",
    )
    .run_commands(
        # Clone MultiTalk repo
        "cd /root && git clone https://github.com/MeiGen-AI/MultiTalk.git",
        # Install any additional requirements
        "cd /root/MultiTalk && pip install -r requirements.txt || true",
    )
    .env({"PYTHONPATH": "/root/MultiTalk"})
)

# Model names as specified in Colab
MODELS = {
    "base": "Wan-AI/Wan2.1-I2V-14B-480P",
    "wav2vec": "TencentGameMate/chinese-wav2vec2-base", 
    "multitalk": "MeiGen-AI/MeiGen-MultiTalk",
}

@app.function(
    image=multitalk_image,
    gpu="a100",  # As specified in Colab
    volumes={
        "/models": model_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
    timeout=900,  # 15 minutes for model download
)
def download_models():
    """Download all required models to persistent volume."""
    import os
    from huggingface_hub import snapshot_download
    import shutil
    
    # Ensure HF token is available
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    
    print("Downloading models...")
    
    # Download each model
    for model_type, repo_id in MODELS.items():
        local_dir = f"/models/{model_type}"
        print(f"\nDownloading {repo_id} to {local_dir}...")
        
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            token=hf_token,
            resume_download=True,
        )
    
    # Set up MultiTalk weights as per Colab
    print("\nSetting up MultiTalk weights...")
    base_index = "/models/base/diffusion_pytorch_model.safetensors.index.json"
    if os.path.exists(base_index):
        shutil.move(base_index, f"{base_index}_old")
    
    shutil.copy(
        "/models/multitalk/diffusion_pytorch_model.safetensors.index.json",
        "/models/base/"
    )
    shutil.copy(
        "/models/multitalk/multitalk.safetensors",
        "/models/base/"
    )
    
    # Persist changes
    model_volume.commit()
    
    print("\nModel download complete!")
    return {"status": "success", "models": list(MODELS.keys())}

@app.function(
    image=multitalk_image,
    gpu="a100",  # Colab uses A100
    volumes={
        "/models": model_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("aws-secret"),
    ],
)
def generate_video(
    audio_data: bytes,
    image_data: bytes, 
    prompt: str = "A person is speaking",
    sample_steps: int = 20,
    output_name: str = "output",
):
    """Generate talking head video using MultiTalk."""
    import sys
    import os
    import json
    import subprocess
    
    sys.path.insert(0, "/root/MultiTalk")
    
    # Save uploaded files to temp locations
    audio_path = "/tmp/input_audio.wav"
    image_path = "/tmp/input_image.jpg"
    
    with open(audio_path, "wb") as f:
        f.write(audio_data)
    
    with open(image_path, "wb") as f:
        f.write(image_data)
    
    # GPU VRAM parameters from Colab
    GPU_VRAM_PARAMS = {
        "NVIDIA A100": 11000000000,
        "NVIDIA A100-SXM4-40GB": 11000000000,
        "NVIDIA A100-SXM4-80GB": 22000000000,
    }
    
    # Detect GPU and set VRAM param
    import torch
    gpu_name = torch.cuda.get_device_name(0)
    vram_param = GPU_VRAM_PARAMS.get(gpu_name, 11000000000)
    
    # Create input JSON
    input_data = {
        "prompt": prompt,
        "cond_image": image_path,
        "cond_audio": {
            "person1": audio_path
        }
    }
    
    input_json_path = "/tmp/input.json"
    with open(input_json_path, "w") as f:
        json.dump(input_data, f)
    
    # Run MultiTalk generation
    cmd = [
        "python3", "/root/MultiTalk/generate_multitalk.py",
        "--ckpt_dir", "/models/base",
        "--wav2vec_dir", "/models/wav2vec",
        "--input_json", input_json_path,
        "--sample_steps", str(sample_steps),
        "--num_persistent_param_in_dit", str(vram_param),
        "--mode", "streaming",
        "--use_teacache",
        "--save_file", output_name,
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise RuntimeError(f"Generation failed: {result.stderr}")
    
    # Return the generated video path
    output_path = f"{output_name}.mp4"
    if os.path.exists(output_path):
        with open(output_path, "rb") as f:
            video_data = f.read()
        return video_data
    else:
        raise FileNotFoundError(f"Output video not found at {output_path}")

@app.function(image=multitalk_image, gpu="t4")  # Test on cheaper GPU
def test_environment():
    """Test that the environment is set up correctly."""
    import torch
    import sys
    import os
    
    # Print results directly to avoid serialization issues
    print("="*60)
    print("Environment Test Results")
    print("="*60)
    
    print(f"\nPython version: {sys.version.split()[0]}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    
    # Check key packages
    print("\nPackage versions:")
    packages = ["torch", "transformers", "xformers", "diffusers", "librosa", "moviepy"]
    for pkg in packages:
        try:
            module = __import__(pkg)
            version = getattr(module, "__version__", "installed")
            print(f"  {pkg}: {version}")
        except Exception as e:
            print(f"  {pkg}: ERROR - {str(e)[:50]}")
    
    # Check MultiTalk
    multitalk_exists = os.path.exists("/root/MultiTalk")
    print(f"\nMultiTalk repo: {'✅ Found' if multitalk_exists else '❌ Not found'}")
    
    if multitalk_exists:
        # Check key files
        files = ["generate_multitalk.py", "requirements.txt"]
        for f in files:
            exists = os.path.exists(f"/root/MultiTalk/{f}")
            print(f"  {f}: {'✅' if exists else '❌'}")
    
    print("\n" + "="*60)
    
    # Return simple serializable data
    return {"status": "complete"}

@app.local_entrypoint()
def main(
    action: str = "test",
    audio_path: str = None,
    image_path: str = None,
    prompt: str = "A person is speaking",
):
    """
    Main entry point for the Modal app.
    
    Actions:
    - test: Test the environment
    - download: Download models
    - generate: Generate video (requires audio_path and image_path)
    """
    if action == "test":
        print("Testing environment...")
        result = test_environment.remote()
        print(f"\nTest completed: {result['status']}")
    
    elif action == "download":
        print("Downloading models...")
        result = download_models.remote()
        print(f"\nDownload result: {result}")
    
    elif action == "generate":
        if not audio_path or not image_path:
            print("Error: generate requires --audio-path and --image-path")
            return
        
        print(f"Generating video...")
        print(f"  Audio: {audio_path}")
        print(f"  Image: {image_path}")
        print(f"  Prompt: {prompt}")
        
        # Read files
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        video_data = generate_video.remote(audio_data, image_data, prompt)
        
        # Save output
        output_path = "output_video.mp4"
        with open(output_path, "wb") as f:
            f.write(video_data)
        
        print(f"\nVideo saved to: {output_path}")
    
    else:
        print(f"Unknown action: {action}")
        print("Valid actions: test, download, generate")

if __name__ == "__main__":
    main()