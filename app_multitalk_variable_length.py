#!/usr/bin/env python3
"""
MeiGen-MultiTalk with proper variable-length video support.
Based on actual MultiTalk capabilities, not artificial 81-frame limitation.
"""

import modal
import os

modal.enable_output()

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

# Complete image with all dependencies
multitalk_complete_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["git", "ffmpeg", "wget"])
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1", 
        "torchaudio==2.4.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "xformers==0.0.28",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers==4.49.0",
        "accelerate",
        "diffusers>=0.30.0",
        "librosa",
        "moviepy",
        "opencv-python",
        "numpy==1.24.4",
        "numba==0.59.1",
        "scipy",
        "soundfile",
        "boto3",
        "huggingface_hub",
        "einops",
        "omegaconf",
        "tqdm",
        "peft",
        "optimum-quanto==0.2.6",
        "easydict",
        "ftfy",
        "pyloudnorm",
        "scikit-image",
        "Pillow",
        "misaki[en]",
    )
    .run_commands(
        "cd /root && git clone https://github.com/MeiGen-AI/MultiTalk.git",
        "cd /root/MultiTalk && pip install -r requirements.txt || true",
    )
    .env({"PYTHONPATH": "/root/MultiTalk"})
)

app = modal.App("multitalk-variable-length")

# Volumes for model storage
model_volume = modal.Volume.from_name("multitalk-models", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("multitalk-hf-cache", create_if_missing=True)

# Model configuration
MODELS = {
    "base": "Wan-AI/Wan2.1-I2V-14B-480P",
    "wav2vec": "TencentGameMate/chinese-wav2vec2-base",
    "multitalk": "MeiGen-AI/MeiGen-MultiTalk",
}

@app.function(
    image=multitalk_complete_image,
    gpu="a100-40gb",
    volumes={
        "/models": model_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("aws-secret"),
    ],
    timeout=1800,  # 30 minutes
)
def download_and_setup_models():
    """
    Download and properly set up all MultiTalk models.
    This needs to run before inference.
    """
    import os
    import shutil
    from huggingface_hub import snapshot_download
    
    print("="*60)
    print("Downloading and Setting Up MultiTalk Models")
    print("="*60)
    
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    
    try:
        # Download each model
        for model_type, repo_id in MODELS.items():
            local_dir = f"/models/{model_type}"
            print(f"\nDownloading {repo_id}...")
            
            # Check if already downloaded
            if os.path.exists(os.path.join(local_dir, "config.json")):
                print(f"  ✅ {model_type} already cached")
                continue
            
            print(f"  Downloading to {local_dir}...")
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                token=hf_token,
                resume_download=True,
            )
            print(f"  ✅ Downloaded {repo_id}")
        
        # Critical: Set up MultiTalk weights in base model directory
        print("\n🔧 Setting up MultiTalk weights...")
        
        base_dir = "/models/base"
        multitalk_dir = "/models/multitalk"
        
        # Check if setup already done
        if os.path.exists(os.path.join(base_dir, "multitalk.safetensors")):
            print("  ✅ MultiTalk weights already set up")
        else:
            # Backup original index file
            original_index = os.path.join(base_dir, "diffusion_pytorch_model.safetensors.index.json")
            backup_index = f"{original_index}_old"
            
            if os.path.exists(original_index) and not os.path.exists(backup_index):
                shutil.move(original_index, backup_index)
                print("  ✅ Backed up original index")
            
            # Copy MultiTalk files
            multitalk_index = os.path.join(multitalk_dir, "diffusion_pytorch_model.safetensors.index.json")
            multitalk_weights = os.path.join(multitalk_dir, "multitalk.safetensors")
            
            if os.path.exists(multitalk_index):
                shutil.copy(multitalk_index, base_dir)
                print("  ✅ Copied MultiTalk index")
            
            if os.path.exists(multitalk_weights):
                shutil.copy(multitalk_weights, base_dir)
                print("  ✅ Copied MultiTalk weights")
        
        # Commit changes to volume
        model_volume.commit()
        print("\n✅ Model setup complete!")
        
        return {"success": True, "message": "All models downloaded and configured"}
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.function(
    image=multitalk_complete_image,
    gpu="a100-40gb",
    volumes={
        "/models": model_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("aws-secret"),
    ],
    timeout=1800,
)
def generate_multitalk_video_smart(
    prompt: str = "A person is speaking enthusiastically",
    image_key: str = "multi1.png",
    audio_key: str = "1.wav",
    sample_steps: int = 20,
    upload_output: bool = True,
    output_prefix: str = "outputs/"
):
    """
    Generate MultiTalk video with smart frame count calculation.
    Handles variable-length videos properly.
    """
    import boto3
    import tempfile
    import shutil
    import os
    import torch
    import sys
    import json
    import subprocess
    import librosa
    from datetime import datetime
    
    print("="*60)
    print("MeiGen-MultiTalk Smart Variable-Length Generation")
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
    
    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nGPU: {gpu_name} ({gpu_memory:.1f} GB)")
    
    try:
        # Setup
        work_dir = tempfile.mkdtemp(prefix="multitalk_")
        s3 = boto3.client('s3')
        
        # Download inputs
        print("\nDownloading from S3...")
        image_path = os.path.join(work_dir, "input.png")
        audio_path = os.path.join(work_dir, "input.wav")
        
        s3.download_file(bucket_name, image_key, image_path)
        s3.download_file(bucket_name, audio_key, audio_path)
        print(f"  ✅ Downloaded files")
        
        # Analyze audio for smart frame calculation
        print("\n🎵 Analyzing audio...")
        y, sr = librosa.load(audio_path, sr=None)
        duration = len(y) / sr
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Sample rate: {sr} Hz")
        
        # Smart frame count calculation based on ACTUAL audio duration
        fps = 25  # MultiTalk's native FPS
        raw_frames = int(duration * fps)
        
        # CRITICAL: Match frame count to actual audio duration
        # Don't force 81 frames if audio is shorter!
        
        if raw_frames < 30:
            # Very short audio - pad to minimum viakBle
            frame_count = 45  # ~1.8s minimum
            print(f"  Audio very short, using minimum {frame_count} frames")
        elif raw_frames <= 81:
            # Use calculated frames for short audio
            frame_count = raw_frames
            # Prefer odd numbers for stability
            if frame_count % 2 == 0:
                frame_count += 1
            print(f"  Using {frame_count} frames (matching {duration:.2f}s audio)")
        elif raw_frames <= 201:
            # Medium length - use calculated frames
            frame_count = raw_frames
            if frame_count % 2 == 0:
                frame_count += 1
            print(f"  Using {frame_count} frames (medium-length video)")
        else:
            # Long video - cap at maximum stable
            frame_count = 201
            print(f"  Capping at 201 frames (long video from {duration:.2f}s audio)")
        
        expected_duration = frame_count / fps
        print(f"  Expected output: {expected_duration:.2f}s ({frame_count} frames @ {fps}fps)")
        
        # GPU VRAM settings
        vram_settings = {
            "NVIDIA A100-SXM4-40GB": 11000000000,
            "NVIDIA A100-SXM4-80GB": 22000000000,
            "NVIDIA A100": 11000000000,
            "NVIDIA A10G": 8000000000,
            "Tesla T4": 5000000000,
        }
        vram_param = vram_settings.get(gpu_name, 11000000000)
        
        # Create input JSON
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
        
        # Run MultiTalk with proper parameters
        print(f"\n🎬 Running MultiTalk inference...")
        print(f"  Frame count: {frame_count}")
        print(f"  Sample steps: {sample_steps}")
        print(f"  VRAM param: {vram_param}")
        
        cmd = [
            "python3", "/root/MultiTalk/generate_multitalk.py",
            "--ckpt_dir", "/models/base",
            "--wav2vec_dir", "/models/wav2vec",
            "--input_json", input_json_path,
            "--frame_num", str(frame_count),  # Variable frame count
            "--sample_steps", str(sample_steps),
            "--num_persistent_param_in_dit", str(vram_param),
            "--mode", "streaming",  # Good for longer videos
            "--use_teacache",  # Acceleration
            "--save_file", output_path,
        ]
        
        print(f"\nCommand: {' '.join(cmd)}")
        
        # Add MultiTalk to path and run
        sys.path.insert(0, "/root/MultiTalk")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=work_dir)
        
        if result.returncode != 0:
            print(f"\n❌ Generation failed!")
            print(f"STDERR: {result.stderr[:1500]}")
            print(f"STDOUT: {result.stdout[:1500]}")
            return {
                "success": False,
                "error": result.stderr,
                "stdout": result.stdout,
                "frame_count": frame_count,
                "audio_duration": duration
            }
        
        # Find output video
        video_path = f"{output_path}.mp4"
        if not os.path.exists(video_path):
            # Look for any mp4 file
            mp4_files = [f for f in os.listdir(work_dir) if f.endswith('.mp4')]
            if mp4_files:
                video_path = os.path.join(work_dir, mp4_files[0])
                print(f"  Found video: {os.path.basename(video_path)}")
            else:
                return {
                    "success": False,
                    "error": "No output video found",
                    "work_dir_contents": os.listdir(work_dir)
                }
        
        video_size = os.path.getsize(video_path)
        print(f"\n✅ Generated video: {os.path.basename(video_path)}")
        print(f"   Size: {video_size:,} bytes")
        
        # Upload to S3 if requested
        if upload_output:
            print("\nUploading to S3...")
            s3_key = f"{output_prefix}multitalk_{timestamp}_{frame_count}f.mp4"
            s3.upload_file(video_path, bucket_name, s3_key)
            
            s3_uri = f"s3://{bucket_name}/{s3_key}"
            print(f"  ✅ Uploaded to: {s3_uri}")
            
            # Clean up
            shutil.rmtree(work_dir)
            
            return {
                "success": True,
                "status": "completed",
                "s3_output": s3_uri,
                "s3_key": s3_key,
                "video_size": video_size,
                "frame_count": frame_count,
                "audio_duration": duration,
                "expected_duration": expected_duration,
                "gpu": gpu_name
            }
        else:
            return {
                "success": True,
                "status": "completed_local",
                "local_output": video_path,
                "video_size": video_size,
                "frame_count": frame_count,
                "audio_duration": duration,
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


if __name__ == "__main__":
    with app.run():
        print("MeiGen-MultiTalk Variable-Length Video Generation\n")
        
        # First ensure models are downloaded and set up
        print("Setting up models...")
        model_result = download_and_setup_models.remote()
        
        if not model_result.get("success"):
            print(f"❌ Model setup failed: {model_result.get('error')}")
            exit(1)
        
        print("✅ Models ready!")
        
        # Now run inference
        print("\n" + "="*60)
        print("Running video generation...")
        
        result = generate_multitalk_video_smart.remote(
            prompt="A person is speaking enthusiastically about AI and technology",
            image_key="multi1.png",
            audio_key="1.wav",
            sample_steps=20,
            upload_output=True
        )
        
        print("\n" + "="*60)
        if result.get("success"):
            print("✅ SUCCESS!")
            print(f"Output: {result.get('s3_output', result.get('local_output'))}")
            print(f"Video size: {result.get('video_size', 0):,} bytes")
            print(f"Frames generated: {result.get('frame_count')}")
            print(f"Audio duration: {result.get('audio_duration', 0):.2f}s")
            print(f"Expected video duration: {result.get('expected_duration', 0):.2f}s")
            print(f"GPU used: {result.get('gpu')}")
        else:
            print("❌ FAILED!")
            print(f"Error: {result.get('error')}")
            if result.get('frame_count'):
                print(f"Attempted frames: {result['frame_count']}")
            if result.get('audio_duration'):
                print(f"Audio duration: {result['audio_duration']:.2f}s")
        print("="*60)
