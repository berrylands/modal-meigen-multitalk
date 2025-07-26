#!/usr/bin/env python3
"""
MeiGen-MultiTalk Real Inference with S3 Integration
Combines S3 input/output with actual MultiTalk inference.
"""

import modal
import os

modal.enable_output()

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

# Create volumes for model storage
model_volume = modal.Volume.from_name("multitalk-models", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("multitalk-hf-cache", create_if_missing=True)

# Build the complete image with all dependencies
multitalk_inference_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install([
        "git", 
        "ffmpeg",
        "libsm6",
        "libxext6", 
        "libxrender-dev",
        "libgomp1",
        "wget",
        "build-essential"
    ])
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
        # Core dependencies from Colab
        "transformers==4.49.0",
        "huggingface_hub",
        "accelerate",
        "diffusers>=0.30.0",
        "librosa",
        "moviepy",
        "opencv-python",
        "numpy==1.24.4",  # Numba 0.59.1 requires numpy < 1.26
        "einops",
        "omegaconf",
        "tqdm",
        "peft",
        "optimum-quanto==0.2.6",
        "easydict",
        "ftfy",
        "pyloudnorm",
        "scikit-image",
        "scipy",
        "soundfile",
        "numba==0.59.1",
        "boto3",  # For S3
        "Pillow",
        "misaki[en]",  # G2P engine needed by MultiTalk
    )
    .run_commands(
        # Clone MultiTalk repo
        "cd /root && git clone https://github.com/MeiGen-AI/MultiTalk.git",
        # Install any additional requirements
        "cd /root/MultiTalk && pip install -r requirements.txt || true",
    )
    .env({
        "PYTHONPATH": "/root/MultiTalk",
        "TORCH_CUDA_ARCH_LIST": "7.0;7.5;8.0;8.6;8.9;9.0",
    })
)

app = modal.App("multitalk-s3-inference")

# Model configuration
MODELS = {
    "base": "Wan-AI/Wan2.1-I2V-14B-480P",
    "wav2vec": "TencentGameMate/chinese-wav2vec2-base", 
    "multitalk": "MeiGen-AI/MeiGen-MultiTalk",
}

@app.function(
    image=multitalk_inference_image,
    gpu="a100-40gb",  # Use A100 for better performance
    volumes={
        "/models": model_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("aws-secret"),
    ],
    timeout=1800,  # 30 minutes for download + inference
)
def generate_multitalk_video_s3(
    prompt: str = "A person is speaking in a professional setting",
    image_key: str = "multi1.png",
    audio_key: str = "1.wav",
    sample_steps: int = 20,
    upload_output: bool = True,
    output_prefix: str = "outputs/"
):
    """
    Generate MultiTalk video using S3 inputs and actual inference.
    """
    import boto3
    import tempfile
    import shutil
    from datetime import datetime
    import torch
    import sys
    import os
    import json
    import subprocess
    from huggingface_hub import snapshot_download
    
    print("="*60)
    print("MeiGen-MultiTalk Real Inference with S3")
    print("="*60)
    
    # Get bucket
    bucket_name = os.environ.get('AWS_BUCKET_NAME')
    if not bucket_name:
        return {"error": "AWS_BUCKET_NAME not found"}
    
    print(f"\nConfiguration:")
    print(f"  Bucket: {bucket_name}")
    print(f"  Image: {image_key}")
    print(f"  Audio: {audio_key}")
    print(f"  Prompt: {prompt}")
    print(f"  Sample steps: {sample_steps}")
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n✅ GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        return {"error": "No GPU available!"}
    
    try:
        # Create work directory
        work_dir = tempfile.mkdtemp(prefix="multitalk_")
        print(f"\nWork directory: {work_dir}")
        
        # Initialize S3
        s3 = boto3.client('s3')
        
        # Download inputs from S3
        print("\nDownloading inputs from S3...")
        
        image_path = os.path.join(work_dir, "input_image.png")
        s3.download_file(bucket_name, image_key, image_path)
        print(f"  ✅ Image: {os.path.getsize(image_path):,} bytes")
        
        audio_path = os.path.join(work_dir, "input_audio.wav")
        s3.download_file(bucket_name, audio_key, audio_path)
        print(f"  ✅ Audio: {os.path.getsize(audio_path):,} bytes")
        
        # Ensure models are downloaded
        print("\nChecking/downloading models...")
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        
        for model_type, repo_id in MODELS.items():
            local_dir = f"/models/{model_type}"
            if not os.path.exists(os.path.join(local_dir, "config.json")):
                print(f"  Downloading {repo_id}...")
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=local_dir,
                    token=hf_token,
                    resume_download=True,
                )
            else:
                print(f"  ✅ {model_type} model already cached")
        
        # Set up MultiTalk weights
        base_index = "/models/base/diffusion_pytorch_model.safetensors.index.json"
        if os.path.exists(base_index) and not os.path.exists(f"{base_index}_old"):
            shutil.move(base_index, f"{base_index}_old")
        
        if not os.path.exists("/models/base/multitalk.safetensors"):
            shutil.copy(
                "/models/multitalk/diffusion_pytorch_model.safetensors.index.json",
                "/models/base/"
            )
            shutil.copy(
                "/models/multitalk/multitalk.safetensors",
                "/models/base/"
            )
        
        # Persist model changes
        model_volume.commit()
        
        # GPU VRAM parameters
        GPU_VRAM_PARAMS = {
            "NVIDIA A100": 11000000000,
            "NVIDIA A100-SXM4-40GB": 11000000000,
            "NVIDIA A100-SXM4-80GB": 22000000000,
            "Tesla T4": 5000000000,
            "NVIDIA A10G": 8000000000,
        }
        
        vram_param = GPU_VRAM_PARAMS.get(gpu_name, 11000000000)
        
        # Create input JSON for MultiTalk
        input_data = {
            "prompt": prompt,
            "cond_image": image_path,
            "cond_audio": {
                "person1": audio_path
            }
        }
        
        input_json_path = os.path.join(work_dir, "input.json")
        with open(input_json_path, "w") as f:
            json.dump(input_data, f)
        
        # Output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"multitalk_{timestamp}"
        output_path = os.path.join(work_dir, output_name)
        
        # Run MultiTalk generation
        print("\nRunning MultiTalk inference...")
        cmd = [
            "python3", "/root/MultiTalk/generate_multitalk.py",
            "--ckpt_dir", "/models/base",
            "--wav2vec_dir", "/models/wav2vec",
            "--input_json", input_json_path,
            "--sample_steps", str(sample_steps),
            "--num_persistent_param_in_dit", str(vram_param),
            "--mode", "streaming",
            "--use_teacache",
            "--save_file", output_path,
        ]
        
        print(f"Command: {' '.join(cmd)}")
        
        # Add MultiTalk to Python path
        sys.path.insert(0, "/root/MultiTalk")
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=work_dir)
        
        if result.returncode != 0:
            print(f"\n❌ Generation failed!")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            
            # Try to diagnose the issue
            if "misaki" in result.stderr.lower():
                print("\n⚠️  Note: 'misaki' package might be missing. Installing...")
                subprocess.run(["pip", "install", "misaki[en]"], check=True)
                # Retry the command
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=work_dir)
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": result.stderr,
                    "stdout": result.stdout
                }
        
        # Check for output video
        video_path = f"{output_path}.mp4"
        if not os.path.exists(video_path):
            # Check work directory for any mp4 files
            mp4_files = [f for f in os.listdir(work_dir) if f.endswith('.mp4')]
            if mp4_files:
                video_path = os.path.join(work_dir, mp4_files[0])
                print(f"\n⚠️  Found video at: {video_path}")
            else:
                return {
                    "success": False,
                    "error": "No output video found",
                    "work_dir_contents": os.listdir(work_dir)
                }
        
        video_size = os.path.getsize(video_path)
        print(f"\n✅ Generated video: {os.path.basename(video_path)} ({video_size:,} bytes)")
        
        # Upload to S3 if requested
        if upload_output:
            print("\nUploading to S3...")
            s3_key = f"{output_prefix}{os.path.basename(video_path)}"
            s3.upload_file(video_path, bucket_name, s3_key)
            
            s3_uri = f"s3://{bucket_name}/{s3_key}"
            print(f"  ✅ Uploaded to: {s3_uri}")
            
            # Clean up
            shutil.rmtree(work_dir)
            print("\n✅ Cleaned up temporary files")
            
            return {
                "success": True,
                "status": "completed",
                "s3_output": s3_uri,
                "s3_key": s3_key,
                "video_size": video_size,
                "gpu": gpu_name
            }
        else:
            return {
                "success": True,
                "status": "completed_local",
                "local_output": video_path,
                "video_size": video_size,
                "gpu": gpu_name
            }
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@app.function(
    image=multitalk_inference_image,
    gpu="t4",  # Cheaper GPU for testing
)
def test_multitalk_setup():
    """Test that MultiTalk is properly set up."""
    import os
    import sys
    import torch
    import subprocess
    
    print("="*60)
    print("MultiTalk Setup Test")
    print("="*60)
    
    # GPU info
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Check MultiTalk
    multitalk_path = "/root/MultiTalk"
    print(f"\nMultiTalk path: {multitalk_path}")
    print(f"Exists: {os.path.exists(multitalk_path)}")
    
    if os.path.exists(multitalk_path):
        # Check key files
        key_files = [
            "generate_multitalk.py",
            "requirements.txt",
            "kokoro/pipeline.py",
        ]
        
        for f in key_files:
            path = os.path.join(multitalk_path, f)
            print(f"  {f}: {'✅' if os.path.exists(path) else '❌'}")
    
    # Try importing
    print("\nTrying imports...")
    sys.path.insert(0, multitalk_path)
    
    try:
        # Check if generate_multitalk.py has help
        result = subprocess.run(
            ["python3", f"{multitalk_path}/generate_multitalk.py", "--help"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✅ generate_multitalk.py --help works!")
        else:
            print(f"❌ Error: {result.stderr[:200]}")
            
            # Check for missing dependencies
            if "misaki" in result.stderr:
                print("\nInstalling misaki...")
                subprocess.run(["pip", "install", "misaki[en]"], check=True)
                print("✅ Installed misaki")
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    return {"status": "complete"}


if __name__ == "__main__":
    with app.run():
        print("MeiGen-MultiTalk S3 Inference\n")
        
        # First test the setup
        print("Testing MultiTalk setup...\n")
        test_result = test_multitalk_setup.remote()
        
        if test_result["status"] != "complete":
            print("❌ Setup test failed!")
            exit(1)
        
        print("\n" + "="*60)
        print("Running inference with S3 inputs...\n")
        
        # Run actual inference
        result = generate_multitalk_video_s3.remote(
            prompt="A person is speaking enthusiastically about AI technology",
            image_key="multi1.png",
            audio_key="1.wav",
            sample_steps=20,
            upload_output=True
        )
        
        print("\n" + "="*60)
        if result.get("success"):
            print("✅ INFERENCE SUCCESSFUL!")
            print(f"Output: {result.get('s3_output', result.get('local_output'))}")
            print(f"Size: {result.get('video_size', 0):,} bytes")
            print(f"GPU: {result.get('gpu')}")
        else:
            print("❌ INFERENCE FAILED!")
            print(f"Error: {result.get('error')}")
            if result.get('stdout'):
                print(f"\nSTDOUT:\n{result['stdout'][:500]}")
            if result.get('traceback'):
                print(f"\nTraceback:\n{result['traceback'][:500]}")
        print("="*60)
