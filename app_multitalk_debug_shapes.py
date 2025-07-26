#!/usr/bin/env python3
"""
MeiGen-MultiTalk with shape debugging and architecture-aligned frame counts.
"""

import modal
import os

modal.enable_output()

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

# Use the working image
working_image = (
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

app = modal.App("multitalk-shape-debug")

model_volume = modal.Volume.from_name("multitalk-models", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("multitalk-hf-cache", create_if_missing=True)

@app.function(
    image=working_image,
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
def debug_and_generate():
    """
    Debug shape issues and find working frame counts.
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
    print("MultiTalk Shape Debugging")
    print("="*60)
    
    bucket_name = os.environ.get('AWS_BUCKET_NAME')
    if not bucket_name:
        return {"error": "AWS_BUCKET_NAME not found"}
    
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"
    print(f"GPU: {gpu_name}")
    
    try:
        work_dir = tempfile.mkdtemp(prefix="multitalk_debug_")
        s3 = boto3.client('s3')
        
        # Download
        print("\nDownloading from S3...")
        image_path = os.path.join(work_dir, "input.png")
        raw_audio_path = os.path.join(work_dir, "raw.wav")
        
        s3.download_file(bucket_name, "multi1.png", image_path)
        s3.download_file(bucket_name, "1.wav", raw_audio_path)
        
        # Audio preprocessing
        print("\nAudio preprocessing...")
        y, sr = librosa.load(raw_audio_path, sr=None)
        duration = len(y) / sr
        print(f"Original: {duration:.2f}s @ {sr}Hz")
        
        # Resample to 16kHz
        y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)
        
        # Try different frame counts that might work with model architecture
        # The error suggests a tensor shape issue with dimensions
        # Common working frame counts: 21, 41, 45, 81, 121, 161, 201
        
        # Let's try 81 frames first (the Colab default)
        frame_count = 81
        target_duration = frame_count / 24.0  # 24 FPS
        target_samples = int(target_duration * 16000)
        
        print(f"\nTrying {frame_count} frames (standard Colab default)")
        print(f"Target duration: {target_duration:.2f}s")
        
        # Adjust audio to match target duration
        if len(y_16k) < target_samples:
            # Pad with silence
            padding = target_samples - len(y_16k)
            y_final = np.pad(y_16k, (0, padding), mode='constant')
            print(f"Padded audio from {len(y_16k)/16000:.2f}s to {target_duration:.2f}s")
        else:
            # Truncate
            y_final = y_16k[:target_samples]
            print(f"Truncated audio from {len(y_16k)/16000:.2f}s to {target_duration:.2f}s")
        
        # Save processed audio
        processed_audio_path = os.path.join(work_dir, "processed.wav")
        sf.write(processed_audio_path, y_final, 16000, subtype='PCM_16')
        
        # Create input JSON
        input_data = {
            "prompt": "A person is speaking",
            "cond_image": image_path,
            "cond_audio": {
                "person1": processed_audio_path
            }
        }
        
        input_json_path = os.path.join(work_dir, "input.json")
        with open(input_json_path, "w") as f:
            json.dump(input_data, f)
        
        # Try with different settings
        print("\n🔧 Trying inference with debugging enabled...")
        
        output_path = os.path.join(work_dir, "output")
        
        # Add environment variable to get more debug info
        env = os.environ.copy()
        env["CUDA_LAUNCH_BLOCKING"] = "1"  # Better error messages
        
        cmd = [
            "python3", "/root/MultiTalk/generate_multitalk.py",
            "--ckpt_dir", "/models/base",
            "--wav2vec_dir", "/models/wav2vec",
            "--input_json", input_json_path,
            "--frame_num", str(frame_count),
            "--sample_steps", "10",  # Fewer steps for debugging
            "--num_persistent_param_in_dit", "11000000000",
            "--mode", "streaming",
            "--use_teacache",
            "--save_file", output_path,
        ]
        
        print(f"Command: {' '.join(cmd)}")
        
        sys.path.insert(0, "/root/MultiTalk")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=work_dir, env=env)
        
        if result.returncode != 0:
            print(f"\n❌ First attempt failed")
            
            # Parse the error for clues
            if "shape" in result.stderr and "invalid" in result.stderr:
                print("\n🔍 Analyzing shape error...")
                # Extract the problematic shapes
                import re
                shape_match = re.search(r"shape '\[(.*?)\]' is invalid for input of size (\d+)", result.stderr)
                if shape_match:
                    bad_shape = shape_match.group(1)
                    tensor_size = int(shape_match.group(2))
                    print(f"Problematic shape: [{bad_shape}]")
                    print(f"Tensor size: {tensor_size}")
                    
                    # Calculate what might work
                    # The shape [1, 11, 4, 56, 112] suggests:
                    # batch=1, something=11, channels=4, height=56, width=112
                    # Total should be: 1 * X * 4 * 56 * 112 = tensor_size
                    expected_dim = tensor_size // (1 * 4 * 56 * 112)
                    print(f"Expected dimension: {expected_dim} (instead of 11)")
                    
                    # Try different frame counts
                    # Common divisors that might work: 21, 41, 81, 121, 161
                    alternate_frames = [21, 41, 45, 121, 161]
                    
                    for alt_frame in alternate_frames:
                        if alt_frame == frame_count:
                            continue
                            
                        print(f"\n🔄 Trying {alt_frame} frames...")
                        
                        # Prepare audio for new frame count
                        alt_duration = alt_frame / 24.0
                        alt_samples = int(alt_duration * 16000)
                        
                        if len(y_16k) < alt_samples:
                            y_alt = np.pad(y_16k, (0, alt_samples - len(y_16k)), mode='constant')
                        else:
                            y_alt = y_16k[:alt_samples]
                        
                        alt_audio_path = os.path.join(work_dir, f"audio_{alt_frame}f.wav")
                        sf.write(alt_audio_path, y_alt, 16000, subtype='PCM_16')
                        
                        # Update input JSON
                        input_data["cond_audio"]["person1"] = alt_audio_path
                        with open(input_json_path, "w") as f:
                            json.dump(input_data, f)
                        
                        # Try again with new frame count
                        cmd[8] = str(alt_frame)  # Update frame_num argument
                        
                        result = subprocess.run(cmd, capture_output=True, text=True, cwd=work_dir, env=env)
                        
                        if result.returncode == 0:
                            print(f"✅ SUCCESS with {alt_frame} frames!")
                            frame_count = alt_frame
                            break
                        else:
                            print(f"❌ Failed with {alt_frame} frames")
                            if "shape" in result.stderr:
                                new_shape = re.search(r"shape '\[(.*?)\]'", result.stderr)
                                if new_shape:
                                    print(f"   Shape error: [{new_shape.group(1)}]")
            
            if result.returncode != 0:
                # Still failing - return detailed error
                return {
                    "success": False,
                    "error": "Shape mismatch persists",
                    "stderr": result.stderr[-2000:],
                    "tried_frames": [81, 21, 41, 45, 121, 161],
                    "audio_preprocessed": True
                }
        
        # Check for output
        video_path = f"{output_path}.mp4"
        if os.path.exists(video_path):
            video_size = os.path.getsize(video_path)
            print(f"\n🎉 SUCCESS! Video generated: {video_size:,} bytes")
            
            # Upload to S3
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            s3_key = f"outputs/multitalk_debug_{timestamp}_{frame_count}f.mp4"
            s3.upload_file(video_path, bucket_name, s3_key)
            
            s3_uri = f"s3://{bucket_name}/{s3_key}"
            print(f"✅ Uploaded: {s3_uri}")
            
            shutil.rmtree(work_dir)
            
            return {
                "success": True,
                "s3_output": s3_uri,
                "video_size": video_size,
                "working_frame_count": frame_count,
                "gpu": gpu_name
            }
        else:
            return {
                "success": False,
                "error": "No output video",
                "work_dir_contents": os.listdir(work_dir)
            }
            
    except Exception as e:
        print(f"\n❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    with app.run():
        print("MultiTalk Shape Debugging\n")
        
        result = debug_and_generate.remote()
        
        print("\n" + "="*60)
        if result.get("success"):
            print("🎉 INFERENCE SUCCESSFUL!")
            print(f"Output: {result['s3_output']}")
            print(f"Video size: {result['video_size']:,} bytes")
            print(f"Working frame count: {result['working_frame_count']}")
            print(f"GPU: {result['gpu']}")
        else:
            print("❌ Failed to find working configuration")
            print(f"Error: {result.get('error')}")
            if result.get('tried_frames'):
                print(f"Tried frame counts: {result['tried_frames']}")
        print("="*60)
