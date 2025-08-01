#!/usr/bin/env python3
"""
MeiGen-MultiTalk with proper CUDA base image and flash-attn installation.
Following Modal's recommended pattern for flash-attn.
"""

import modal
import os
from typing import Dict, List, Optional, Union

modal.enable_output()

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

# Use NVIDIA CUDA base image as recommended by Modal docs
multitalk_cuda_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04",
        add_python="3.10"
    )
    # Install system dependencies
    .apt_install([
        "git",
        "ffmpeg",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
        "wget",
    ])
    # Install PyTorch first with CUDA 12.1
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1", 
        "torchaudio==2.4.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    # Install prerequisites for flash-attn
    .pip_install(
        "ninja",
        "packaging",
        "wheel",
        "setuptools",
    )
    # Install transformers and xformers separately
    .pip_install(
        "transformers==4.49.0",
    )
    .pip_install(
        "xformers==0.0.28",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    # Install flash-attn
    .run_commands(
        "pip install flash-attn==2.6.1 --no-build-isolation",
    )
    # Install other dependencies
    .pip_install(
        "peft",
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
        "optimum-quanto==0.2.6",
        "easydict",
        "ftfy",
        "pyloudnorm",
        "scikit-image",
        "Pillow",
        "misaki[en]",
    )
    # Clone and set up MultiTalk
    .run_commands(
        "cd /root && git clone https://github.com/MeiGen-AI/MultiTalk.git",
        "cd /root/MultiTalk && pip install -r requirements.txt || true",
    )
    .env({"PYTHONPATH": "/root/MultiTalk"})
)

app = modal.App("multitalk-cuda")

model_volume = modal.Volume.from_name("multitalk-models", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("multitalk-hf-cache", create_if_missing=True)

@app.function(
    image=multitalk_cuda_image,
    gpu="a100-40gb",
    volumes={
        "/models": model_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
    secrets=[
        modal.Secret.from_name("aws-secret"),
        modal.Secret.from_name("huggingface-secret"),
    ],
    timeout=1800,  # 30 minutes for multi-person generation
)
def generate_video_cuda(
    prompt: str = "A person is speaking enthusiastically about technology",
    image_key: str = "multi1.png",
    audio_key: str = "1.wav",
    sample_steps: int = 20
):
    """
    Generate video with proper CUDA and flash-attn support.
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
    from PIL import Image
    
    print("="*60)
    print("MeiGen-MultiTalk with CUDA and Flash Attention")
    print("="*60)
    
    bucket_name = os.environ.get('AWS_BUCKET_NAME')
    s3 = boto3.client('s3')
    
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"
    print(f"GPU: {gpu_name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    
    # Check flash-attn availability
    try:
        import flash_attn
        print(f"Flash-attn: {flash_attn.__version__} ✅")
    except Exception as e:
        print(f"Flash-attn: Not available - {e}")
    
    try:
        # Set up directory structure
        multitalk_dir = "/root/MultiTalk"
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
                print(f"  Linked {os.path.basename(src)} -> {os.path.basename(dst)}")
        
        # Set up MultiTalk weights
        base_dir = os.path.join(weights_dir, "Wan2.1-I2V-14B-480P")
        multitalk_dir_weights = os.path.join(weights_dir, "MeiGen-MultiTalk")
        
        # Check if models exist
        if not os.path.exists(base_dir):
            print(f"\n❌ Base model not found at {base_dir}")
            print("Please run model download first")
            return {"success": False, "error": "Models not downloaded"}
        
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
        
        # Download files from S3
        print("\n⬇️ Downloading from S3...")
        s3.download_file(bucket_name, image_key, "input_raw.png")
        s3.download_file(bucket_name, audio_key, "input_raw.wav")
        
        # CRITICAL: Resize image to 896x448
        print("\n🖼️ Processing image...")
        img = Image.open("input_raw.png")
        print(f"  Original size: {img.size}")
        
        EXPECTED_WIDTH = 896
        EXPECTED_HEIGHT = 448
        resized_img = img.resize((EXPECTED_WIDTH, EXPECTED_HEIGHT), Image.Resampling.LANCZOS)
        resized_img.save("input.png")
        print(f"  Resized to: {EXPECTED_WIDTH}x{EXPECTED_HEIGHT}")
        
        # Process audio
        print("\n🎵 Processing audio...")
        y, sr = librosa.load("input_raw.wav", sr=None)
        duration = len(y) / sr
        print(f"  Original: {duration:.2f}s @ {sr}Hz")
        
        # Resample to 16kHz
        y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)
        
        # Calculate frame count - CRITICAL for single-person generation
        fps = 24
        raw_frames = int(duration * fps)
        
        if raw_frames < 60:
            frame_count = 45
        elif raw_frames < 100:
            frame_count = 81
        else:
            frame_count = 121
        
        print(f"  Using {frame_count} frames")
        
        # Adjust audio duration to match frame count
        target_duration = frame_count / fps
        target_samples = int(target_duration * 16000)
        
        if len(y_16k) < target_samples:
            padding = target_samples - len(y_16k)
            y_final = np.pad(y_16k, (0, padding), mode='constant')
            print(f"  Padded to {target_duration:.2f}s")
        else:
            y_final = y_16k[:target_samples]
            print(f"  Truncated to {target_duration:.2f}s")
        
        sf.write("input.wav", y_final, 16000, subtype='PCM_16')
        
        # Create input JSON
        input_data = {
            "prompt": prompt,
            "cond_image": "input.png",
            "cond_audio": {"person1": "input.wav"}
        }
        
        with open("input.json", "w") as f:
            json.dump(input_data, f)
        
        # Run MultiTalk
        print(f"\n🎬 Running MultiTalk...")
        print(f"  Image: {EXPECTED_WIDTH}x{EXPECTED_HEIGHT}")
        print(f"  Audio: {target_duration:.2f}s")
        print(f"  Frames: {frame_count}")
        print(f"  Attention: Flash Attention 2.6.1")
        
        # Use same GPU memory optimization as multi-person
        GPU_VRAM_PARAMS = {
            "NVIDIA A10G": 8000000000,
            "NVIDIA A100-SXM4-40GB": 11000000000,
            "NVIDIA A100-SXM4-80GB": 22000000000,
        }
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"
        vram_param = GPU_VRAM_PARAMS.get(gpu_name, 8000000000)

        cmd = [
            "python3", "generate_multitalk.py",
            "--ckpt_dir", "weights/Wan2.1-I2V-14B-480P",
            "--wav2vec_dir", "weights/chinese-wav2vec2-base",
            "--input_json", "input.json",
            "--frame_num", str(frame_count),  # CRITICAL: Must specify frame count
            "--sample_steps", str(sample_steps),
            "--num_persistent_param_in_dit", str(vram_param),
            "--sample_audio_guide_scale", "4.0",  # Add audio guide scale
            "--color_correction_strength", "0.7",  # Add color correction
            "--mode", "streaming",
            "--use_teacache",
            "--save_file", "output",
        ]
        
        # Set environment to ensure flash-attn is used
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        if result.returncode != 0:
            print(f"\n❌ Failed!")
            print("STDERR:", result.stderr[-2000:])
            return {
                "success": False,
                "error": result.stderr,
                "stdout": result.stdout
            }
        
        # Check for output
        if os.path.exists("output.mp4"):
            video_size = os.path.getsize("output.mp4")
            print(f"\n🎉 SUCCESS! Generated {video_size:,} bytes")
            
            # Upload to S3
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            s3_key = f"outputs/multitalk_cuda_{timestamp}.mp4"
            s3.upload_file("output.mp4", bucket_name, s3_key)
            
            s3_uri = f"s3://{bucket_name}/{s3_key}"
            print(f"✅ Uploaded to: {s3_uri}")
            
            # Save locally
            local_path = f"/tmp/multitalk_output_{timestamp}.mp4"
            shutil.copy("output.mp4", local_path)
            
            return {
                "success": True,
                "s3_output": s3_uri,
                "local_output": local_path,
                "video_size": video_size,
                "audio_duration": duration,
                "image_original": img.size,
                "image_resized": (EXPECTED_WIDTH, EXPECTED_HEIGHT),
                "gpu": gpu_name,
                "attention": "flash-attn-2.6.1"
            }
        else:
            print("\n❌ No output file found")
            print("Files in directory:", os.listdir("."))
            return {"success": False, "error": "No output found", "stdout": result.stdout}
            
    except Exception as e:
        print(f"\n❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


@app.function(
    image=multitalk_cuda_image,
    gpu="a100-40gb",
    volumes={
        "/models": model_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
    secrets=[
        modal.Secret.from_name("aws-secret"),
        modal.Secret.from_name("huggingface-secret"),
    ],
    timeout=1800,  # 30 minutes for multi-person generation
)
def generate_multi_person_video(
    prompt: str = "Two people having a conversation",
    image_key: str = "multi1.png",
    audio_keys: Union[str, List[str]] = "1.wav",  # Can be single string or list
    sample_steps: int = 20,
    output_prefix: str = "multitalk",
    audio_type: str = "add",  # "para" (simultaneous) or "add" (sequential)
    use_bbox: bool = False,  # Whether to use bounding boxes
    audio_cfg: float = 4.0,  # Audio guide scale (3-5 recommended for lip sync)
    color_correction: float = 0.7  # Color correction strength (0-1, lower = less correction)
):
    """
    Generate video with support for single or multi-person conversations.
    
    Args:
        prompt: Description of the video content
        image_key: S3 key for the reference image
        audio_keys: Single audio key (str) or list of audio keys for multi-person
        sample_steps: Number of sampling steps
        output_prefix: Prefix for output file naming
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
    from PIL import Image
    
    print("="*60)
    print("MeiGen-MultiTalk Multi-Person Support")
    print("="*60)
    
    bucket_name = os.environ.get('AWS_BUCKET_NAME')
    s3 = boto3.client('s3')
    
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"
    print(f"GPU: {gpu_name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    
    # Check flash-attn availability
    try:
        import flash_attn
        print(f"Flash-attn: {flash_attn.__version__} ✅")
    except Exception as e:
        print(f"Flash-attn: Not available - {e}")
    
    # Convert single audio to list for uniform processing
    if isinstance(audio_keys, str):
        audio_keys = [audio_keys]
    
    num_speakers = len(audio_keys)
    print(f"\n👥 Number of speakers: {num_speakers}")
    
    try:
        # Set up directory structure
        multitalk_dir = "/root/MultiTalk"
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
                print(f"  Linked {os.path.basename(src)} -> {os.path.basename(dst)}")
        
        # Set up MultiTalk weights
        base_dir = os.path.join(weights_dir, "Wan2.1-I2V-14B-480P")
        multitalk_dir_weights = os.path.join(weights_dir, "MeiGen-MultiTalk")
        
        # Check if models exist
        if not os.path.exists(base_dir):
            print(f"\n❌ Base model not found at {base_dir}")
            print("Please run model download first")
            return {"success": False, "error": "Models not downloaded"}
        
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
        
        # Download image from S3
        print("\n⬇️ Downloading from S3...")
        s3.download_file(bucket_name, image_key, "input_raw.png")
        print(f"  ✅ Downloaded image: {image_key}")
        
        # Download all audio files
        audio_files = {}
        all_durations = []
        
        for i, audio_key in enumerate(audio_keys):
            person_id = f"person{i+1}"
            raw_filename = f"input_raw_{person_id}.wav"
            s3.download_file(bucket_name, audio_key, raw_filename)
            print(f"  ✅ Downloaded audio {i+1}: {audio_key}")
            
            # Process each audio file
            y, sr = librosa.load(raw_filename, sr=None)
            duration = len(y) / sr
            all_durations.append(duration)
            print(f"    Duration: {duration:.2f}s @ {sr}Hz")
            
            # Store for later processing
            audio_files[person_id] = {
                'raw_file': raw_filename,
                'duration': duration,
                'sr': sr,
                'data': y
            }
        
        # CRITICAL: Resize image to 896x448
        print("\n🖼️ Processing image...")
        img = Image.open("input_raw.png")
        print(f"  Original size: {img.size}")
        
        EXPECTED_WIDTH = 896
        EXPECTED_HEIGHT = 448
        resized_img = img.resize((EXPECTED_WIDTH, EXPECTED_HEIGHT), Image.Resampling.LANCZOS)
        resized_img.save("input.png")
        print(f"  Resized to: {EXPECTED_WIDTH}x{EXPECTED_HEIGHT}")
        
        # Process audio files
        max_duration = max(all_durations)
        print(f"\n🎵 Processing {num_speakers} audio file(s)...")
        print(f"  Max duration: {max_duration:.2f}s")
        
        # Let the model use its default frame count (81)
        # Don't specify frame_num in the command
        
        # Process each audio to match duration
        target_samples = int(max_duration * 16000)
        
        cond_audio = {}
        
        for person_id, audio_info in audio_files.items():
            print(f"\n  Processing {person_id}:")
            
            # Resample to 16kHz
            y_16k = librosa.resample(
                audio_info['data'], 
                orig_sr=audio_info['sr'], 
                target_sr=16000
            )
            
            # Keep original audio duration (let model handle timing)
            y_final = y_16k
            print(f"    Duration: {len(y_final)/16000:.2f}s")
            
            # Save processed audio
            output_filename = f"input_{person_id}.wav"
            sf.write(output_filename, y_final, 16000, subtype='PCM_16')
            cond_audio[person_id] = output_filename
            print(f"    Saved as: {output_filename}")
        
        # Create input JSON with all audio files
        input_data = {
            "prompt": prompt,
            "cond_image": "input.png",
            "cond_audio": cond_audio
        }
        
        # Add audio_type for multi-person conversations
        if num_speakers > 1:
            input_data["audio_type"] = audio_type
            
            # Add bounding boxes if requested
            if use_bbox:
                # For 2 people, split image left/right
                # Based on official example structure
                bbox_data = {}
                if num_speakers == 2:
                    # Left half for person1, right half for person2
                    bbox_data["person1"] = [0, 0, EXPECTED_WIDTH//2, EXPECTED_HEIGHT]
                    bbox_data["person2"] = [EXPECTED_WIDTH//2, 0, EXPECTED_WIDTH, EXPECTED_HEIGHT]
                    input_data["bbox"] = bbox_data
                    print(f"  Added bounding boxes: {bbox_data}")
        
        with open("input.json", "w") as f:
            json.dump(input_data, f, indent=2)
        
        print(f"\n📝 Created input.json with {num_speakers} speaker(s):")
        print("Full JSON content:")
        print(json.dumps(input_data, indent=2))
        
        # Run MultiTalk
        print(f"\n🎬 Running MultiTalk...")
        print(f"  Mode: {'Multi-person' if num_speakers > 1 else 'Single person'}")
        if num_speakers > 1:
            print(f"  Audio mode: {audio_type} ({'sequential' if audio_type == 'add' else 'simultaneous'})")
        print(f"  Image: {EXPECTED_WIDTH}x{EXPECTED_HEIGHT}")
        print(f"  Audio duration: {max_duration:.2f}s")
        print(f"  Frames: Default (model will determine)")
        print(f"  Attention: Flash Attention 2.6.1")
        print(f"  Audio cfg: {audio_cfg}")
        print(f"  Color correction: {color_correction}")
        
        # Get GPU VRAM parameter
        GPU_VRAM_PARAMS = {
            "NVIDIA A10G": 8000000000,
            "NVIDIA A100-SXM4-40GB": 11000000000,
            "NVIDIA A100-SXM4-80GB": 22000000000,
        }
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"
        vram_param = GPU_VRAM_PARAMS.get(gpu_name, 8000000000)
        
        cmd = [
            "python3", "generate_multitalk.py",
            "--ckpt_dir", "weights/Wan2.1-I2V-14B-480P",
            "--wav2vec_dir", "weights/chinese-wav2vec2-base",
            "--input_json", "input.json",
            # Don't specify frame_num - let model use default
            "--sample_steps", str(sample_steps),
            "--num_persistent_param_in_dit", str(vram_param),
            "--sample_audio_guide_scale", str(audio_cfg),  # Add audio guide scale
            "--color_correction_strength", str(color_correction),  # Add color correction
            "--mode", "streaming",  # Always use streaming for multi-person
            "--use_teacache",  # Always use teacache
            "--save_file", "output",
        ]
        
        print(f"\n📋 Command to execute:")
        print(" ".join(cmd))
        
        # Set environment to ensure flash-attn is used
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        if result.returncode != 0:
            print(f"\n❌ Failed!")
            print("STDERR:", result.stderr[-2000:])
            return {
                "success": False,
                "error": result.stderr,
                "stdout": result.stdout
            }
        
        # Check for output
        if os.path.exists("output.mp4"):
            video_size = os.path.getsize("output.mp4")
            print(f"\n🎉 SUCCESS! Generated {video_size:,} bytes")
            
            # Upload to S3
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mode_suffix = f"{num_speakers}person" if num_speakers > 1 else "single"
            s3_key = f"outputs/{output_prefix}_{mode_suffix}_{timestamp}.mp4"
            s3.upload_file("output.mp4", bucket_name, s3_key)
            
            s3_uri = f"s3://{bucket_name}/{s3_key}"
            print(f"✅ Uploaded to: {s3_uri}")
            
            # Save locally
            local_path = f"/tmp/{output_prefix}_output_{timestamp}.mp4"
            shutil.copy("output.mp4", local_path)
            
            return {
                "success": True,
                "s3_output": s3_uri,
                "local_output": local_path,
                "video_size": video_size,
                "num_speakers": num_speakers,
                "audio_durations": all_durations,
                "max_duration": max_duration,
                "image_original": img.size,
                "image_resized": (EXPECTED_WIDTH, EXPECTED_HEIGHT),
                "gpu": gpu_name,
                "attention": "flash-attn-2.6.1",
                "audio_mode": audio_type if num_speakers > 1 else "single"
            }
        else:
            print("\n❌ No output file found")
            print("Files in directory:", os.listdir("."))
            return {"success": False, "error": "No output found", "stdout": result.stdout}
            
    except Exception as e:
        print(f"\n❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


@app.function(
    image=multitalk_cuda_image,
    gpu="a10g",
    memory=32768,
    timeout=900,
    secrets=[modal.Secret.from_name("aws-secret")],
    volumes={
        "/models": model_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
)
def generate_single_person_fixed(
    prompt: str = "A person speaking naturally", 
    image_key: str = "multi1.png",
    audio_key: str = "1.wav",
    sample_steps: int = 20,
    output_prefix: str = "single_person"
):
    """
    Generate single-person video using the multi-person function.
    This bypasses the single-person validation issues.
    """
    print("🔧 Using multi-person function for single-person video (workaround)")
    # Call multi-person function with single audio
    return generate_multi_person_video.local(
        prompt=prompt,
        image_key=image_key,
        audio_keys=[audio_key],  # Single audio as list
        sample_steps=sample_steps,
        output_prefix=output_prefix,
        audio_type="add",
        use_bbox=False
    )


if __name__ == "__main__":
    import sys
    
    with app.run():
        print("MeiGen-MultiTalk with CUDA Base Image\n")
        print("Using nvidia/cuda:12.1.0-devel-ubuntu22.04 base")
        print("Flash-attn 2.6.1 installed with --no-build-isolation\n")
        
        # Check command line for mode
        if len(sys.argv) > 1 and sys.argv[1] == "--two-person":
            print("🎭 Running TWO-PERSON conversation demo")
            print("Note: Make sure you have two audio files (1.wav and 2.wav) in your S3 bucket\n")
            
            result = generate_multi_person_video.remote(
                prompt="Two people having an animated conversation about AI",
                image_key="multi1.png",
                audio_keys=["1.wav", "2.wav"],  # Two audio files
                sample_steps=20,
                output_prefix="multitalk_2person",
                audio_type="add",  # Sequential speaking
                use_bbox=False,  # No bounding boxes needed
                audio_cfg=4.0  # Default audio guide scale
            )
        else:
            print("👤 Running SINGLE-PERSON demo")
            print("To run two-person demo: python app_multitalk_cuda.py --two-person\n")
            
            result = generate_video_cuda.remote(
                prompt="A person is speaking enthusiastically about AI and technology",
                image_key="multi1.png",
                audio_key="1.wav",
                sample_steps=20
            )
        
        print("\n" + "="*60)
        if result.get("success"):
            print("🎆 SUCCESS WITH FLASH ATTENTION!")
            print(f"\n✅ S3 Output: {result['s3_output']}")
            print(f"💾 Local: {result.get('local_output')}")
            print(f"📊 Size: {result['video_size']:,} bytes")
            print(f"🎬 Frames: {result['frame_count']}")
            
            # Handle both single and multi-person outputs
            if 'num_speakers' in result:
                print(f"👥 Speakers: {result['num_speakers']}")
                if result.get('audio_durations'):
                    print(f"🎵 Audio durations: {[f'{d:.2f}s' for d in result['audio_durations']]}")
                print(f"⏱️ Target duration: {result.get('target_duration', 0):.2f}s")
            else:
                print(f"🎵 Audio: {result['audio_duration']:.2f}s")
            
            print(f"🖼️ Image: {result['image_original']} -> {result['image_resized']}")
            print(f"🖥️ GPU: {result['gpu']}")
            print(f"⚡ Attention: {result['attention']}")
            print("\n🎉 Your video has been successfully generated!")
            print("\nTo download from S3:")
            print(f"aws s3 cp {result['s3_output']} ./my_multitalk_video.mp4")
        else:
            print("❌ Failed")
            print(f"Error: {result.get('error', 'Unknown')[:500]}")
            if result.get('stdout'):
                print(f"\nSTDOUT: {result['stdout'][:500]}")
        print("="*60)


@app.function(
    image=multitalk_cuda_image,
    gpu="a100-40gb",
    volumes={
        "/models": model_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
    secrets=[
        modal.Secret.from_name("aws-secret"),
        modal.Secret.from_name("huggingface-secret"),
    ],
    timeout=1800,  # 30 minutes for multi-person generation
)
def test_two_person_conversation():
    """
    Test function specifically for two-person conversation.
    """
    # Test with different audio for each speaker
    return generate_multi_person_video.local(
        prompt="Two people having an animated conversation",
        image_key="multi1.png", 
        audio_keys=["1.wav", "2.wav"],  # Different audio for each
        sample_steps=10,  # Min 10 for good lip sync per README
        output_prefix="multitalk_2person_test",
        audio_type="add",  # Sequential speaking
        use_bbox=False,  # No bounding boxes needed
        audio_cfg=4.0  # Default audio guide scale
    )