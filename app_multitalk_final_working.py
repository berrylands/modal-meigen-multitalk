#!/usr/bin/env python3
"""
MeiGen-MultiTalk with proper Colab-style audio preprocessing.
Based on actual Colab implementation insights.
"""

import modal
import os

modal.enable_output()

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

# Complete image matching Colab requirements
colab_compatible_image = (
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
        "transformers==4.49.0",  # Exact Colab version
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

app = modal.App("multitalk-colab-compatible")

model_volume = modal.Volume.from_name("multitalk-models", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("multitalk-hf-cache", create_if_missing=True)

MODELS = {
    "base": "Wan-AI/Wan2.1-I2V-14B-480P",
    "wav2vec": "TencentGameMate/chinese-wav2vec2-base",
    "multitalk": "MeiGen-AI/MeiGen-MultiTalk",
}

@app.function(
    image=colab_compatible_image,
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
def setup_models_colab_style():
    """
    Download and set up models exactly like Colab.
    """
    import os
    import shutil
    from huggingface_hub import snapshot_download
    
    print("Setting up models (Colab-style)...")
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    
    # Download models if not cached
    for model_type, repo_id in MODELS.items():
        local_dir = f"/models/{model_type}"
        if not os.path.exists(os.path.join(local_dir, "config.json")):
            print(f"Downloading {repo_id}...")
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                token=hf_token,
                resume_download=True,
            )
    
    # Critical Colab setup: Copy MultiTalk weights into base model
    base_dir = "/models/base"
    multitalk_dir = "/models/multitalk"
    
    # Backup and replace index file (Colab style)
    original_index = os.path.join(base_dir, "diffusion_pytorch_model.safetensors.index.json")
    backup_index = f"{original_index}_old"
    
    if os.path.exists(original_index) and not os.path.exists(backup_index):
        shutil.move(original_index, backup_index)
    
    # Copy MultiTalk files
    shutil.copy(
        os.path.join(multitalk_dir, "diffusion_pytorch_model.safetensors.index.json"),
        base_dir
    )
    shutil.copy(
        os.path.join(multitalk_dir, "multitalk.safetensors"),
        base_dir
    )
    
    model_volume.commit()
    return {"success": True}


@app.function(
    image=colab_compatible_image,
    gpu="a100-40gb",
    volumes={
        "/models": model_volume,
    },
    secrets=[
        modal.Secret.from_name("aws-secret"),
    ],
    timeout=1200,
)
def generate_with_colab_audio_processing(
    prompt: str = "A person is speaking enthusiastically",
    image_key: str = "multi1.png",
    audio_key: str = "1.wav",
    sample_steps: int = 20
):
    """
    Generate video with proper Colab-style audio preprocessing.
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
    print("MultiTalk with Colab-Style Audio Processing")
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
        image_path = os.path.join(work_dir, "input_image.png")
        raw_audio_path = os.path.join(work_dir, "raw_audio.wav")
        
        s3.download_file(bucket_name, image_key, image_path)
        s3.download_file(bucket_name, audio_key, raw_audio_path)
        print(f"Downloaded: {image_key}, {audio_key}")
        
        # CRITICAL: Colab-style audio preprocessing
        print("\n🎵 Colab-style audio preprocessing...")
        
        # Step 1: Load audio with librosa
        y_orig, sr_orig = librosa.load(raw_audio_path, sr=None)
        orig_duration = len(y_orig) / sr_orig
        print(f"Original: {orig_duration:.2f}s @ {sr_orig}Hz")
        
        # Step 2: Resample to 16kHz (MANDATORY for wav2vec2)
        TARGET_SR = 16000  # Colab requirement
        y_16k = librosa.resample(y_orig, orig_sr=sr_orig, target_sr=TARGET_SR)
        new_duration = len(y_16k) / TARGET_SR
        print(f"Resampled: {new_duration:.2f}s @ {TARGET_SR}Hz")
        
        # Step 3: Calculate optimal frame count (Colab approach)
        FPS = 24  # Colab uses 24 FPS
        calculated_frames = int(new_duration * FPS)
        
        # Step 4: Choose frame count strategy
        if calculated_frames < 21:
            # Too short - pad audio to minimum duration
            min_duration = 21 / FPS  # ~0.87 seconds
            target_samples = int(min_duration * TARGET_SR)
            
            # Pad audio with silence
            padding_needed = target_samples - len(y_16k)
            y_final = np.pad(y_16k, (0, padding_needed), mode='constant', constant_values=0)
            frame_count = 21
            print(f"Padded short audio to {min_duration:.2f}s ({frame_count} frames)")
            
        elif calculated_frames <= 81:
            # Good length - use as is
            y_final = y_16k
            frame_count = calculated_frames
            # Ensure odd number (MultiTalk prefers odd frames)
            if frame_count % 2 == 0:
                frame_count += 1
            print(f"Using calculated {frame_count} frames")
        
        elif calculated_frames <= 201:
            # Medium length - use calculated
            y_final = y_16k
            frame_count = calculated_frames
            if frame_count % 2 == 0:
                frame_count += 1
            print(f"Using {frame_count} frames (medium video)")
        
        else:
            # Too long - truncate to max stable length
            max_duration = 201 / FPS  # ~8.37 seconds
            max_samples = int(max_duration * TARGET_SR)
            y_final = y_16k[:max_samples]
            frame_count = 201
            print(f"Truncated to {max_duration:.2f}s ({frame_count} frames)")
        
        # Step 5: Save processed audio
        processed_audio_path = os.path.join(work_dir, "processed_audio.wav")
        sf.write(processed_audio_path, y_final, TARGET_SR, subtype='PCM_16')
        
        final_duration = len(y_final) / TARGET_SR
        expected_video_duration = frame_count / FPS
        
        print(f"\n📊 Final audio specs:")
        print(f"  Duration: {final_duration:.2f}s")
        print(f"  Sample rate: {TARGET_SR}Hz")
        print(f"  Samples: {len(y_final):,}")
        print(f"  Expected video: {expected_video_duration:.2f}s ({frame_count} frames)")
        print(f"  Duration match: {abs(final_duration - expected_video_duration) < 0.1}")
        
        # Step 6: Create input JSON
        input_data = {
            "prompt": prompt,
            "cond_image": image_path,
            "cond_audio": {
                "person1": processed_audio_path  # Use processed audio
            }
        }
        
        input_json_path = os.path.join(work_dir, "input.json")
        with open(input_json_path, "w") as f:
            json.dump(input_data, f)
        
        # Step 7: VRAM settings (Colab-style)
        vram_settings = {
            "NVIDIA A100-SXM4-40GB": 11000000000,
            "NVIDIA A100-SXM4-80GB": 22000000000,
            "NVIDIA A100": 11000000000,
        }
        vram_param = vram_settings.get(gpu_name, 11000000000)
        
        # Step 8: Run MultiTalk (exact Colab command)
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
            print(f"STDERR: {result.stderr[:2000]}")
            return {
                "success": False,
                "error": result.stderr,
                "audio_duration": final_duration,
                "frame_count": frame_count,
                "audio_preprocessed": True
            }
        
        # Find output video
        video_path = f"{output_path}.mp4"
        if not os.path.exists(video_path):
            mp4_files = [f for f in os.listdir(work_dir) if f.endswith('.mp4')]
            if mp4_files:
                video_path = os.path.join(work_dir, mp4_files[0])
            else:
                return {"success": False, "error": "No output video found"}
        
        video_size = os.path.getsize(video_path)
        print(f"\n✅ SUCCESS! Generated: {os.path.basename(video_path)} ({video_size:,} bytes)")
        
        # Upload to S3
        print("\nUploading to S3...")
        s3_key = f"outputs/multitalk_colab_{timestamp}_{frame_count}f.mp4"
        s3.upload_file(video_path, bucket_name, s3_key)
        
        s3_uri = f"s3://{bucket_name}/{s3_key}"
        print(f"✅ Uploaded: {s3_uri}")
        
        # Cleanup
        shutil.rmtree(work_dir)
        
        return {
            "success": True,
            "s3_output": s3_uri,
            "video_size": video_size,
            "frame_count": frame_count,
            "original_audio_duration": orig_duration,
            "processed_audio_duration": final_duration,
            "video_duration": expected_video_duration,
            "gpu": gpu_name,
            "colab_preprocessing": True
        }
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    with app.run():
        print("MeiGen-MultiTalk with Colab-Compatible Processing\n")
        
        # Setup models
        print("Setting up models...")
        setup_result = setup_models_colab_style.remote()
        if not setup_result.get("success"):
            print("❌ Model setup failed")
            exit(1)
        print("✅ Models ready")
        
        # Run generation with proper audio preprocessing
        print("\n" + "="*60)
        print("Running generation with Colab-style audio processing...")
        
        result = generate_with_colab_audio_processing.remote(
            prompt="A person is speaking enthusiastically about AI technology",
            image_key="multi1.png",
            audio_key="1.wav",
            sample_steps=20
        )
        
        print("\n" + "="*60)
        if result.get("success"):
            print("🎉 REAL INFERENCE SUCCESS!")
            print(f"Output: {result['s3_output']}")
            print(f"Video size: {result['video_size']:,} bytes")
            print(f"Frames: {result['frame_count']}")
            print(f"Original audio: {result['original_audio_duration']:.2f}s")
            print(f"Processed audio: {result['processed_audio_duration']:.2f}s")
            print(f"Video duration: {result['video_duration']:.2f}s")
            print(f"GPU: {result['gpu']}")
            print("\n✅ Your S3 inputs successfully processed with real MultiTalk inference!")
        else:
            print("❌ FAILED")
            print(f"Error: {result.get('error')}")
            if result.get('audio_preprocessed'):
                print("✅ Audio preprocessing completed successfully")
                print(f"Frame count: {result.get('frame_count')}")
                print(f"Audio duration: {result.get('audio_duration'):.2f}s")
        print("="*60)
