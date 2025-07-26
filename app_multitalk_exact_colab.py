#!/usr/bin/env python3
"""
MeiGen-MultiTalk EXACT Colab implementation.
No frame_num parameter - let the model decide!
"""

import modal
import os

modal.enable_output()

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

# Exact Colab-matching image
colab_exact_image = (
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

app = modal.App("multitalk-exact-colab")

model_volume = modal.Volume.from_name("multitalk-models", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("multitalk-hf-cache", create_if_missing=True)

@app.function(
    image=colab_exact_image,
    gpu="a100-40gb",  # Colab uses A100
    volumes={
        "/models": model_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
    secrets=[
        modal.Secret.from_name("aws-secret"),
    ],
    timeout=900,
)
def generate_exactly_like_colab(
    prompt: str = "A person is speaking enthusiastically about technology",
    image_key: str = "multi1.png",
    audio_key: str = "1.wav",
    sample_steps: int = 20
):
    """
    Generate video EXACTLY like Colab - no frame_num parameter!
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
    from datetime import datetime
    from PIL import Image
    import numpy as np
    
    print("="*60)
    print("MeiGen-MultiTalk EXACT Colab Implementation")
    print("="*60)
    
    bucket_name = os.environ.get('AWS_BUCKET_NAME')
    if not bucket_name:
        return {"error": "AWS_BUCKET_NAME not found"}
    
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"
    print(f"GPU: {gpu_name}")
    
    try:
        work_dir = tempfile.mkdtemp(prefix="multitalk_colab_")
        s3 = boto3.client('s3')
        
        # Download from S3
        print("\nDownloading from S3...")
        raw_image_path = os.path.join(work_dir, "raw_image.png")
        raw_audio_path = os.path.join(work_dir, "raw_audio.wav")
        
        s3.download_file(bucket_name, image_key, raw_image_path)
        s3.download_file(bucket_name, audio_key, raw_audio_path)
        print(f"✅ Downloaded: {image_key}, {audio_key}")
        
        # Check image dimensions
        print("\n🖼️ Checking image...")
        img = Image.open(raw_image_path)
        print(f"  Original size: {img.size}")
        print(f"  Format: {img.format}")
        print(f"  Mode: {img.mode}")
        
        # For 480p output, let's try standard resolutions
        # 480p typically means 854x480 or 640x480
        # But VAE latent space [56, 112] suggests 448x896 or 896x448
        
        # Save image as-is first (Colab might not resize)
        image_path = os.path.join(work_dir, "input.png")
        img.save(image_path, "PNG")
        print(f"  Saved as PNG: {image_path}")
        
        # Audio preprocessing - ONLY resample to 16kHz
        print("\n🎵 Audio preprocessing (Colab-style)...")
        y_orig, sr_orig = librosa.load(raw_audio_path, sr=None)
        orig_duration = len(y_orig) / sr_orig
        print(f"  Original: {orig_duration:.2f}s @ {sr_orig}Hz")
        
        # Resample to 16kHz (MANDATORY)
        TARGET_SR = 16000
        y_16k = librosa.resample(y_orig, orig_sr=sr_orig, target_sr=TARGET_SR)
        
        # Save resampled audio
        audio_path = os.path.join(work_dir, "input.wav")
        sf.write(audio_path, y_16k, TARGET_SR, subtype='PCM_16')
        print(f"  Resampled to 16kHz and saved")
        
        # Create input JSON (exact Colab format)
        input_data = {
            "prompt": prompt,
            "cond_image": image_path,
            "cond_audio": {
                "person1": audio_path
            }
        }
        
        input_json_path = os.path.join(work_dir, "input.json")
        with open(input_json_path, "w") as f:
            json.dump(input_data, f, indent=2)
        
        print("\n📝 Input JSON:")
        print(json.dumps(input_data, indent=2))
        
        # VRAM parameter (Colab-style)
        vram_params = {
            "NVIDIA A100": 11000000000,
            "NVIDIA A100-SXM4-40GB": 11000000000,
            "NVIDIA A100-SXM4-80GB": 22000000000,
        }
        vram_param = vram_params.get(gpu_name, 11000000000)
        
        # Run MultiTalk EXACTLY like Colab (NO frame_num!)
        print("\n🎬 Running MultiTalk (Colab command)...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(work_dir, "output")
        
        # EXACT Colab command - NO frame_num parameter!
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
        
        print(f"\nCommand (exactly like Colab):")
        print(" ".join(cmd))
        print("\nNote: NO --frame_num parameter!")
        
        sys.path.insert(0, "/root/MultiTalk")
        
        # Run with environment for better debugging
        env = os.environ.copy()
        env["CUDA_LAUNCH_BLOCKING"] = "1"
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=work_dir, env=env)
        
        print(f"\nReturn code: {result.returncode}")
        
        if result.returncode != 0:
            print(f"\n❌ Generation failed!")
            
            # Show error details
            if result.stderr:
                print("\nSTDERR:")
                print(result.stderr[-2000:])
            
            if result.stdout:
                print("\nSTDOUT (last part):")
                print(result.stdout[-1000:])
            
            return {
                "success": False,
                "error": "Generation failed",
                "stderr": result.stderr,
                "stdout": result.stdout,
                "colab_exact": True,
                "no_frame_num": True
            }
        
        # Find output video
        video_path = f"{output_path}.mp4"
        if not os.path.exists(video_path):
            # Look for any mp4
            mp4_files = [f for f in os.listdir(work_dir) if f.endswith('.mp4')]
            if mp4_files:
                video_path = os.path.join(work_dir, mp4_files[0])
                print(f"Found video: {mp4_files[0]}")
            else:
                print("\nWork directory contents:")
                for f in os.listdir(work_dir):
                    print(f"  {f}")
                return {
                    "success": False,
                    "error": "No output video found",
                    "work_dir": os.listdir(work_dir)
                }
        
        video_size = os.path.getsize(video_path)
        print(f"\n🎉 SUCCESS! Generated: {os.path.basename(video_path)}")
        print(f"Size: {video_size:,} bytes")
        
        # Get video info
        import cv2
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        print(f"\nVideo info:")
        print(f"  Frames: {frame_count}")
        print(f"  FPS: {fps}")
        print(f"  Resolution: {width}x{height}")
        print(f"  Duration: {frame_count/fps:.2f}s")
        
        # Upload to S3
        print("\nUploading to S3...")
        s3_key = f"outputs/multitalk_colab_{timestamp}.mp4"
        s3.upload_file(video_path, bucket_name, s3_key)
        
        s3_uri = f"s3://{bucket_name}/{s3_key}"
        print(f"✅ Uploaded: {s3_uri}")
        
        # Save locally too
        local_copy = f"./multitalk_colab_{timestamp}.mp4"
        shutil.copy(video_path, local_copy)
        print(f"💾 Saved locally: {local_copy}")
        
        # Cleanup
        shutil.rmtree(work_dir)
        
        return {
            "success": True,
            "s3_output": s3_uri,
            "local_output": local_copy,
            "video_size": video_size,
            "video_info": {
                "frames": frame_count,
                "fps": fps,
                "resolution": f"{width}x{height}",
                "duration": frame_count/fps
            },
            "image_original_size": img.size,
            "audio_duration": orig_duration,
            "gpu": gpu_name,
            "colab_exact": True
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
        print("MeiGen-MultiTalk EXACT Colab Implementation\n")
        print("Key difference: NO --frame_num parameter!\n")
        
        result = generate_exactly_like_colab.remote(
            prompt="A person is speaking enthusiastically about AI and technology",
            image_key="multi1.png",
            audio_key="1.wav",
            sample_steps=20
        )
        
        print("\n" + "="*60)
        if result.get("success"):
            print("🎆 REAL MULTITALK SUCCESS WITH COLAB EXACT IMPLEMENTATION!")
            print(f"\nS3 Output: {result['s3_output']}")
            print(f"Local Output: {result['local_output']}")
            print(f"Video Size: {result['video_size']:,} bytes")
            
            if 'video_info' in result:
                info = result['video_info']
                print(f"\nVideo Details:")
                print(f"  Frames: {info['frames']}")
                print(f"  FPS: {info['fps']}")
                print(f"  Resolution: {info['resolution']}")
                print(f"  Duration: {info['duration']:.2f}s")
            
            print(f"\nOriginal image size: {result.get('image_original_size')}")
            print(f"Audio duration: {result.get('audio_duration', 0):.2f}s")
            print(f"GPU: {result['gpu']}")
            
            print("\n✅ Your video has been generated using EXACT Colab implementation!")
            print("\nTo download:")
            print(f"aws s3 cp {result['s3_output']} ./my_video.mp4")
        else:
            print("❌ Generation Failed")
            print(f"Error: {result.get('error')}")
            print("\nThis used EXACT Colab command (no --frame_num)")
        print("="*60)
