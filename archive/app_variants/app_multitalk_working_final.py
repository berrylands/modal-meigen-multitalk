#!/usr/bin/env python3
"""
MeiGen-MultiTalk with correct 4n+1 frame constraint.
Final working version based on architecture requirements.
"""

import modal
import os

modal.enable_output()

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

# Complete working image
final_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["git", "ffmpeg"])
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

app = modal.App("multitalk-final-working")

model_volume = modal.Volume.from_name("multitalk-models", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("multitalk-hf-cache", create_if_missing=True)

# Valid frame counts that follow the 4n+1 pattern
VALID_FRAME_COUNTS = [21, 45, 81, 121, 161, 201]

@app.function(
    image=final_image,
    gpu="a100-40gb",
    volumes={
        "/models": model_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
    secrets=[
        modal.Secret.from_name("aws-secret"),
    ],
    timeout=900,
)
def generate_multitalk_video_final(
    prompt: str = "A person is speaking enthusiastically about technology",
    image_key: str = "multi1.png",
    audio_key: str = "1.wav",
    sample_steps: int = 20
):
    """
    Final working MultiTalk generation with correct frame constraints.
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
    import soundfile as sf
    import numpy as np
    from datetime import datetime
    
    print("="*60)
    print("MeiGen-MultiTalk Final Working Version")
    print("="*60)
    
    bucket_name = os.environ.get('AWS_BUCKET_NAME')
    if not bucket_name:
        return {"error": "AWS_BUCKET_NAME not found"}
    
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"
    print(f"GPU: {gpu_name}")
    
    try:
        work_dir = tempfile.mkdtemp(prefix="multitalk_final_")
        s3 = boto3.client('s3')
        
        # Download from S3
        print("\nDownloading from S3...")
        image_path = os.path.join(work_dir, "input.png")
        raw_audio_path = os.path.join(work_dir, "raw.wav")
        
        s3.download_file(bucket_name, image_key, image_path)
        s3.download_file(bucket_name, audio_key, raw_audio_path)
        print(f"✅ Downloaded: {image_key}, {audio_key}")
        
        # Audio preprocessing with 16kHz resampling
        print("\n🎵 Audio preprocessing...")
        y_orig, sr_orig = librosa.load(raw_audio_path, sr=None)
        orig_duration = len(y_orig) / sr_orig
        print(f"Original: {orig_duration:.2f}s @ {sr_orig}Hz")
        
        # CRITICAL: Resample to 16kHz
        TARGET_SR = 16000
        y_16k = librosa.resample(y_orig, orig_sr=sr_orig, target_sr=TARGET_SR)
        duration_16k = len(y_16k) / TARGET_SR
        
        # Calculate nearest valid frame count (4n+1 pattern)
        FPS = 24  # Standard FPS
        raw_frames = int(duration_16k * FPS)
        
        # Find nearest valid frame count
        print("\n🎬 Selecting valid frame count...")
        print(f"Raw frame calculation: {raw_frames} frames")
        
        # Choose the nearest valid frame count
        if raw_frames <= VALID_FRAME_COUNTS[0]:
            frame_count = VALID_FRAME_COUNTS[0]  # 21 minimum
        elif raw_frames >= VALID_FRAME_COUNTS[-1]:
            frame_count = VALID_FRAME_COUNTS[-1]  # 201 maximum
        else:
            # Find nearest valid count
            differences = [abs(raw_frames - valid) for valid in VALID_FRAME_COUNTS]
            min_idx = differences.index(min(differences))
            frame_count = VALID_FRAME_COUNTS[min_idx]
        
        print(f"Selected frame count: {frame_count} (follows 4n+1 pattern)")
        print(f"Pattern check: ({frame_count} - 1) / 4 = {(frame_count - 1) / 4}")
        
        # Adjust audio to match selected frame count
        target_duration = frame_count / FPS
        target_samples = int(target_duration * TARGET_SR)
        
        if len(y_16k) < target_samples:
            # Pad with silence
            padding = target_samples - len(y_16k)
            y_final = np.pad(y_16k, (0, padding), mode='constant', constant_values=0)
            print(f"Padded audio to {target_duration:.2f}s")
        else:
            # Truncate
            y_final = y_16k[:target_samples]
            print(f"Truncated audio to {target_duration:.2f}s")
        
        # Save processed audio
        processed_audio_path = os.path.join(work_dir, "processed.wav")
        sf.write(processed_audio_path, y_final, TARGET_SR, subtype='PCM_16')
        
        print(f"\n📊 Final configuration:")
        print(f"  Audio: {target_duration:.2f}s @ {TARGET_SR}Hz")
        print(f"  Frames: {frame_count} (valid 4n+1)")
        print(f"  Video: {frame_count/FPS:.2f}s @ {FPS}fps")
        
        # Create input JSON
        input_data = {
            "prompt": prompt,
            "cond_image": image_path,
            "cond_audio": {
                "person1": processed_audio_path
            }
        }
        
        input_json_path = os.path.join(work_dir, "input.json")
        with open(input_json_path, "w") as f:
            json.dump(input_data, f)
        
        # VRAM settings
        vram_settings = {
            "NVIDIA A100-SXM4-40GB": 11000000000,
            "NVIDIA A100-SXM4-80GB": 22000000000,
            "NVIDIA A100": 11000000000,
            "NVIDIA A10G": 8000000000,
        }
        vram_param = vram_settings.get(gpu_name, 11000000000)
        
        # Run MultiTalk
        print("\n🎬 Running MultiTalk inference...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(work_dir, f"output_{timestamp}")
        
        cmd = [
            "python3", "/root/MultiTalk/generate_multitalk.py",
            "--ckpt_dir", "/models/base",
            "--wav2vec_dir", "/models/wav2vec",
            "--input_json", input_json_path,
            "--frame_num", str(frame_count),
            "--sample_steps", str(sample_steps),
            "--num_persistent_param_in_dit", str(vram_param),
            "--mode", "streaming",
            "--use_teacache",
            "--save_file", output_path,
        ]
        
        print(f"Command: {' '.join(cmd)}")
        
        sys.path.insert(0, "/root/MultiTalk")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=work_dir)
        
        if result.returncode != 0:
            print(f"\n❌ Generation failed!")
            print(f"Return code: {result.returncode}")
            
            # Show last part of error
            if result.stderr:
                print("\nError details:")
                print(result.stderr[-1500:])
            
            return {
                "success": False,
                "error": "Generation failed",
                "stderr": result.stderr,
                "frame_count": frame_count,
                "audio_duration": target_duration
            }
        
        # Find output video
        video_path = f"{output_path}.mp4"
        if not os.path.exists(video_path):
            # Look for any mp4
            mp4_files = [f for f in os.listdir(work_dir) if f.endswith('.mp4')]
            if mp4_files:
                video_path = os.path.join(work_dir, mp4_files[0])
            else:
                return {
                    "success": False,
                    "error": "No output video found",
                    "work_dir": os.listdir(work_dir)
                }
        
        video_size = os.path.getsize(video_path)
        print(f"\n🎉 SUCCESS! Generated: {os.path.basename(video_path)}")
        print(f"Size: {video_size:,} bytes")
        
        # Upload to S3
        print("\nUploading to S3...")
        s3_key = f"outputs/multitalk_final_{timestamp}_{frame_count}f.mp4"
        s3.upload_file(video_path, bucket_name, s3_key)
        
        s3_uri = f"s3://{bucket_name}/{s3_key}"
        print(f"✅ Uploaded: {s3_uri}")
        
        # Also download locally for verification
        local_copy = f"./output_multitalk_{timestamp}.mp4"
        with open(local_copy, 'wb') as f:
            with open(video_path, 'rb') as src:
                f.write(src.read())
        print(f"💾 Local copy saved: {local_copy}")
        
        # Cleanup
        shutil.rmtree(work_dir)
        
        return {
            "success": True,
            "s3_output": s3_uri,
            "local_output": local_copy,
            "video_size": video_size,
            "frame_count": frame_count,
            "audio_duration": target_duration,
            "gpu": gpu_name
        }
        
    except Exception as e:
        print(f"\n❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


if __name__ == "__main__":
    with app.run():
        print("MeiGen-MultiTalk Final Working Implementation\n")
        
        result = generate_multitalk_video_final.remote(
            prompt="A person is speaking enthusiastically about AI and technology",
            image_key="multi1.png",
            audio_key="1.wav",
            sample_steps=20
        )
        
        print("\n" + "="*60)
        if result.get("success"):
            print("🎆 REAL MULTITALK INFERENCE SUCCESS!")
            print(f"S3 Output: {result['s3_output']}")
            print(f"Local Output: {result.get('local_output', 'N/A')}")
            print(f"Video Size: {result['video_size']:,} bytes")
            print(f"Frame Count: {result['frame_count']} (valid 4n+1)")
            print(f"Audio Duration: {result['audio_duration']:.2f}s")
            print(f"GPU: {result['gpu']}")
            print("\n✅ Your video has been generated and uploaded to S3!")
            print("\nTo download from S3:")
            print(f"aws s3 cp {result['s3_output']} ./my_video.mp4")
        else:
            print("❌ Generation Failed")
            print(f"Error: {result.get('error')}")
            if result.get('frame_count'):
                print(f"Frame count used: {result['frame_count']}")
        print("="*60)
