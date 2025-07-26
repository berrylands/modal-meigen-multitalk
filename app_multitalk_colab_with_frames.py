#!/usr/bin/env python3
"""
MeiGen-MultiTalk with Colab paths + appropriate frame count for short audio.
"""

import modal
import os

modal.enable_output()

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

# Same image as before
colab_image = (
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
        "mkdir -p /content",
        "cd /content && git clone https://github.com/MeiGen-AI/MultiTalk.git",
        "cd /content/MultiTalk && pip install -r requirements.txt || true",
    )
)

app = modal.App("multitalk-colab-frames")

model_volume = modal.Volume.from_name("multitalk-models", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("multitalk-hf-cache", create_if_missing=True)

@app.function(
    image=colab_image,
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
def generate_final_working():
    """
    Generate video with Colab paths AND appropriate frame count.
    """
    import boto3
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
    s3 = boto3.client('s3')
    
    # Set up Colab directory structure
    multitalk_dir = "/content/MultiTalk"
    weights_dir = os.path.join(multitalk_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    # Create symlinks for models
    model_links = [
        ("/models/base", os.path.join(weights_dir, "Wan2.1-I2V-14B-480P")),
        ("/models/wav2vec", os.path.join(weights_dir, "chinese-wav2vec2-base")),
        ("/models/multitalk", os.path.join(weights_dir, "MeiGen-MultiTalk")),
    ]
    
    for src, dst in model_links:
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)
    
    # Set up MultiTalk weights
    base_dir = os.path.join(weights_dir, "Wan2.1-I2V-14B-480P")
    multitalk_dir_weights = os.path.join(weights_dir, "MeiGen-MultiTalk")
    
    index_file = os.path.join(base_dir, "diffusion_pytorch_model.safetensors.index.json")
    if os.path.exists(index_file) and not os.path.exists(f"{index_file}_old"):
        shutil.move(index_file, f"{index_file}_old")
    
    files_to_copy = [
        "diffusion_pytorch_model.safetensors.index.json",
        "multitalk.safetensors"
    ]
    
    for f in files_to_copy:
        src = os.path.join(multitalk_dir_weights, f)
        dst = os.path.join(base_dir, f)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
    
    # Change to MultiTalk directory
    os.chdir(multitalk_dir)
    print(f"Working directory: {os.getcwd()}")
    
    # Download and process files
    print("\nDownloading from S3...")
    s3.download_file(bucket_name, "multi1.png", "input.png")
    s3.download_file(bucket_name, "1.wav", "input_raw.wav")
    
    # Audio preprocessing
    print("\nProcessing audio...")
    y, sr = librosa.load("input_raw.wav", sr=None)
    duration = len(y) / sr
    print(f"  Duration: {duration:.2f}s")
    
    # Resample to 16kHz
    y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)
    
    # Calculate appropriate frame count for this audio
    fps = 24  # Default FPS
    raw_frames = int(duration * fps)
    
    # Use 45 frames for short audio (valid 4n+1 pattern)
    if raw_frames < 60:
        frame_count = 45  # Good for ~1.88s audio
    elif raw_frames < 100:
        frame_count = 81  # Default
    else:
        frame_count = 121  # For longer audio
    
    print(f"  Using {frame_count} frames for {duration:.2f}s audio")
    
    # Adjust audio duration to match frame count
    target_duration = frame_count / fps
    target_samples = int(target_duration * 16000)
    
    if len(y_16k) < target_samples:
        # Pad with silence
        padding = target_samples - len(y_16k)
        y_final = np.pad(y_16k, (0, padding), mode='constant')
        print(f"  Padded audio to {target_duration:.2f}s")
    else:
        # Truncate
        y_final = y_16k[:target_samples]
        print(f"  Truncated audio to {target_duration:.2f}s")
    
    # Save processed audio
    import numpy as np
    sf.write("input.wav", y_final, 16000, subtype='PCM_16')
    
    # Create input JSON
    input_data = {
        "prompt": "A person is speaking",
        "cond_image": "input.png",
        "cond_audio": {"person1": "input.wav"}
    }
    
    with open("input.json", "w") as f:
        json.dump(input_data, f)
    
    # Run MultiTalk with frame count
    print(f"\n🎬 Running MultiTalk with {frame_count} frames...")
    
    cmd = [
        "python3", "generate_multitalk.py",
        "--ckpt_dir", "weights/Wan2.1-I2V-14B-480P",
        "--wav2vec_dir", "weights/chinese-wav2vec2-base",
        "--input_json", "input.json",
        "--frame_num", str(frame_count),  # Add frame count!
        "--sample_steps", "20",
        "--num_persistent_param_in_dit", "11000000000",
        "--mode", "streaming",
        "--use_teacache",
        "--save_file", "output",
    ]
    
    print(" ".join(cmd))
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ Failed!")
        print(result.stderr[-2000:])
        return {"success": False, "error": result.stderr}
    
    # Check for output
    if os.path.exists("output.mp4"):
        video_size = os.path.getsize("output.mp4")
        print(f"\n🎉 SUCCESS! Generated {video_size:,} bytes")
        
        # Upload to S3
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = f"outputs/multitalk_success_{timestamp}_{frame_count}f.mp4"
        s3.upload_file("output.mp4", bucket_name, s3_key)
        
        return {
            "success": True,
            "s3_output": f"s3://{bucket_name}/{s3_key}",
            "video_size": video_size,
            "frame_count": frame_count,
            "audio_duration": duration
        }
    else:
        return {"success": False, "error": "No output found"}


if __name__ == "__main__":
    with app.run():
        print("Running final working version...\n")
        
        result = generate_final_working.remote()
        
        print("\n" + "="*60)
        if result.get("success"):
            print("🎆 REAL MULTITALK SUCCESS!")
            print(f"Output: {result['s3_output']}")
            print(f"Size: {result['video_size']:,} bytes")
            print(f"Frames: {result['frame_count']}")
            print(f"Audio: {result['audio_duration']:.2f}s")
        else:
            print("❌ Failed")
            print(f"Error: {result.get('error', 'Unknown')[:500]}")
        print("="*60)
