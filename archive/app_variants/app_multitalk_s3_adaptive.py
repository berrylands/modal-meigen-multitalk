#!/usr/bin/env python3
"""
MeiGen-MultiTalk with adaptive frame count based on audio duration.
"""

import modal
import os

modal.enable_output()

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

# Use simpler image that builds faster
simple_multitalk_image = (
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
        "Pillow",
        "misaki[en]",
    )
    .run_commands(
        "cd /root && git clone https://github.com/MeiGen-AI/MultiTalk.git",
        "cd /root/MultiTalk && pip install -r requirements.txt || true",
    )
    .env({"PYTHONPATH": "/root/MultiTalk"})
)

app = modal.App("multitalk-adaptive")

# Create/use volumes
model_volume = modal.Volume.from_name("multitalk-models", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("multitalk-hf-cache", create_if_missing=True)

@app.function(
    image=simple_multitalk_image,
    gpu="a10g",  # Use A10G - good balance of performance/cost
    volumes={
        "/models": model_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("aws-secret"),
    ],
    timeout=1200,
)
def generate_adaptive_video():
    """
    Generate video with frame count adapted to audio duration.
    """
    import boto3
    import tempfile
    import os
    import torch
    import sys
    import json
    import subprocess
    import librosa
    import soundfile as sf
    from huggingface_hub import snapshot_download
    import shutil
    
    print("="*60)
    print("MultiTalk Adaptive Frame Generation")
    print("="*60)
    
    # Get bucket
    bucket_name = os.environ.get('AWS_BUCKET_NAME')
    if not bucket_name:
        return {"error": "AWS_BUCKET_NAME not found"}
    
    print(f"\nBucket: {bucket_name}")
    
    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
    
    try:
        # Create work dir
        work_dir = tempfile.mkdtemp(prefix="multitalk_")
        print(f"Work dir: {work_dir}")
        
        # Download from S3
        s3 = boto3.client('s3')
        
        print("\nDownloading from S3...")
        image_path = os.path.join(work_dir, "input.png")
        audio_path = os.path.join(work_dir, "input.wav")
        
        s3.download_file(bucket_name, "multi1.png", image_path)
        s3.download_file(bucket_name, "1.wav", audio_path)
        print("✅ Downloaded files")
        
        # Analyze audio to determine frame count
        print("\n🎵 Analyzing audio...")
        y, sr = librosa.load(audio_path, sr=None)
        duration = len(y) / sr
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Sample rate: {sr} Hz")
        
        # Calculate appropriate frame count
        # MultiTalk typically uses 24-30 FPS
        fps = 24
        frame_count = int(duration * fps)
        # Round to nearest odd number (MultiTalk seems to prefer odd numbers)
        if frame_count % 2 == 0:
            frame_count += 1
        
        print(f"\n🎥 Video parameters:")
        print(f"  FPS: {fps}")
        print(f"  Calculated frames: {frame_count}")
        print(f"  Expected duration: {frame_count/fps:.2f}s")
        
        # Ensure models are available
        print("\n🤖 Checking models...")
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        
        models = {
            "base": "Wan-AI/Wan2.1-I2V-14B-480P",
            "wav2vec": "TencentGameMate/chinese-wav2vec2-base",
            "multitalk": "MeiGen-AI/MeiGen-MultiTalk",
        }
        
        for model_type, repo_id in models.items():
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
                print(f"  ✅ {model_type} model cached")
        
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
        
        model_volume.commit()
        
        # Prepare input JSON
        input_data = {
            "prompt": "A person is speaking",
            "cond_image": image_path,
            "cond_audio": {
                "person1": audio_path
            }
        }
        
        input_json_path = os.path.join(work_dir, "input.json")
        with open(input_json_path, "w") as f:
            json.dump(input_data, f)
        
        # Output path
        output_path = os.path.join(work_dir, "output")
        
        # GPU VRAM settings
        vram_settings = {
            "NVIDIA A10G": 8000000000,
            "Tesla T4": 5000000000,
            "NVIDIA A100": 11000000000,
        }
        vram_param = vram_settings.get(gpu_name, 8000000000)
        
        # Run MultiTalk
        print("\n🎬 Running MultiTalk inference...")
        print(f"  Frame count: {frame_count}")
        print(f"  Sample steps: 20")
        
        cmd = [
            "python3", "/root/MultiTalk/generate_multitalk.py",
            "--ckpt_dir", "/models/base",
            "--wav2vec_dir", "/models/wav2vec",
            "--input_json", input_json_path,
            "--frame_num", str(frame_count),  # Use calculated frame count
            "--sample_steps", "20",
            "--num_persistent_param_in_dit", str(vram_param),
            "--mode", "streaming",
            "--use_teacache",
            "--save_file", output_path,
        ]
        
        sys.path.insert(0, "/root/MultiTalk")
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=work_dir)
        
        if result.returncode != 0:
            print(f"\n❌ Generation failed!")
            print(f"STDERR: {result.stderr[:1000]}")
            print(f"STDOUT: {result.stdout[:1000]}")
            return {
                "success": False,
                "error": result.stderr,
                "stdout": result.stdout
            }
        
        # Find output video
        video_path = f"{output_path}.mp4"
        if not os.path.exists(video_path):
            # Look for any mp4 file
            mp4_files = [f for f in os.listdir(work_dir) if f.endswith('.mp4')]
            if mp4_files:
                video_path = os.path.join(work_dir, mp4_files[0])
            else:
                return {"success": False, "error": "No output video found"}
        
        video_size = os.path.getsize(video_path)
        print(f"\n✅ Generated video: {os.path.basename(video_path)}")
        print(f"   Size: {video_size:,} bytes")
        
        # Upload to S3
        print("\nUploading to S3...")
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = f"outputs/multitalk_{timestamp}_{frame_count}frames.mp4"
        s3.upload_file(video_path, bucket_name, s3_key)
        
        s3_uri = f"s3://{bucket_name}/{s3_key}"
        print(f"✅ Uploaded to: {s3_uri}")
        
        # Clean up
        shutil.rmtree(work_dir)
        
        return {
            "success": True,
            "s3_output": s3_uri,
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
        print("MultiTalk Adaptive Frame Generation\n")
        
        result = generate_adaptive_video.remote()
        
        print("\n" + "="*60)
        if result.get("success"):
            print("✅ SUCCESS!")
            print(f"Output: {result['s3_output']}")
            print(f"Size: {result['video_size']:,} bytes")
            print(f"Frames: {result['frame_count']}")
            print(f"Audio duration: {result['audio_duration']:.2f}s")
            print(f"GPU: {result['gpu']}")
        else:
            print("❌ FAILED!")
            print(f"Error: {result.get('error')}")
        print("="*60)
