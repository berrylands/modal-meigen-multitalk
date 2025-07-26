#!/usr/bin/env python3
"""
MeiGen-MultiTalk with patched attention mechanism.
"""

import modal
import os

modal.enable_output()

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

# Image with attention patch
patched_image = (
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

app = modal.App("multitalk-patched")

model_volume = modal.Volume.from_name("multitalk-models", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("multitalk-hf-cache", create_if_missing=True)

@app.function(
    image=patched_image,
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
def generate_with_patch(
    prompt: str = "A person is speaking enthusiastically about technology",
    image_key: str = "multi1.png",
    audio_key: str = "1.wav",
    sample_steps: int = 20
):
    """
    Generate video with patched attention mechanism.
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
    print("MeiGen-MultiTalk with Patched Attention")
    print("="*60)
    
    bucket_name = os.environ.get('AWS_BUCKET_NAME')
    s3 = boto3.client('s3')
    
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"
    print(f"GPU: {gpu_name}")
    
    try:
        # CRITICAL: Patch the attention.py file to disable flash attention
        attention_file = "/content/MultiTalk/wan/modules/attention.py"
        if os.path.exists(attention_file):
            print("\n🔧 Patching attention mechanism...")
            
            # Read the original file
            with open(attention_file, 'r') as f:
                original_content = f.read()
            
            # Simple patch: just modify the assertion and let existing fallback work
            patched_content = original_content
            
            # Replace the assertion with a simple pass
            if "assert FLASH_ATTN_2_AVAILABLE" in patched_content:
                # Find the line and replace it
                lines = patched_content.split('\n')
                new_lines = []
                for line in lines:
                    if "assert FLASH_ATTN_2_AVAILABLE" in line:
                        # Replace with a check that uses the fallback
                        indent = len(line) - len(line.lstrip())
                        new_lines.append(' ' * indent + "if not FLASH_ATTN_2_AVAILABLE:")
                        new_lines.append(' ' * (indent + 4) + "# Use fallback implementation")
                    else:
                        new_lines.append(line)
                
                patched_content = '\n'.join(new_lines)
            
            # Also force FLASH_ATTN_2_AVAILABLE to False
            if "FLASH_ATTN_2_AVAILABLE = True" in patched_content:
                patched_content = patched_content.replace(
                    "FLASH_ATTN_2_AVAILABLE = True",
                    "FLASH_ATTN_2_AVAILABLE = False"
                )
            
            # Write the patched content
            with open(attention_file, 'w') as f:
                f.write(patched_content)
            
            print("  ✅ Patched attention.py to use fallback")
        
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
        
        # Download files from S3
        print("\n⬇️ Downloading from S3...")
        s3.download_file(bucket_name, image_key, "input_raw.png")
        s3.download_file(bucket_name, audio_key, "input_raw.wav")
        
        # Resize image to 896x448
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
        
        # Calculate frame count
        fps = 24
        raw_frames = int(duration * fps)
        
        if raw_frames < 60:
            frame_count = 45
        elif raw_frames < 100:
            frame_count = 81
        else:
            frame_count = 121
        
        print(f"  Using {frame_count} frames")
        
        # Adjust audio duration
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
        print(f"  Attention: Patched (flash disabled)")
        
        cmd = [
            "python3", "generate_multitalk.py",
            "--ckpt_dir", "weights/Wan2.1-I2V-14B-480P",
            "--wav2vec_dir", "weights/chinese-wav2vec2-base",
            "--input_json", "input.json",
            "--frame_num", str(frame_count),
            "--sample_steps", str(sample_steps),
            "--num_persistent_param_in_dit", "11000000000",
            "--mode", "streaming",
            "--use_teacache",
            "--save_file", "output",
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"\n❌ Failed!")
            print(result.stderr[-2000:])
            return {
                "success": False,
                "error": result.stderr,
                "attention_patched": True
            }
        
        # Check for output
        if os.path.exists("output.mp4"):
            video_size = os.path.getsize("output.mp4")
            print(f"\n🎉 SUCCESS! Generated {video_size:,} bytes")
            
            # Upload to S3
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            s3_key = f"outputs/multitalk_patched_{timestamp}_{frame_count}f.mp4"
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
                "frame_count": frame_count,
                "audio_duration": duration,
                "image_original": img.size,
                "image_resized": (EXPECTED_WIDTH, EXPECTED_HEIGHT),
                "gpu": gpu_name,
                "attention_patched": True
            }
        else:
            return {"success": False, "error": "No output found"}
            
    except Exception as e:
        print(f"\n❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    with app.run():
        print("MeiGen-MultiTalk with Patched Attention\n")
        print("Approach: Patch attention.py to disable flash attention requirement\n")
        
        result = generate_with_patch.remote(
            prompt="A person is speaking enthusiastically about AI and technology",
            image_key="multi1.png",
            audio_key="1.wav",
            sample_steps=20
        )
        
        print("\n" + "="*60)
        if result.get("success"):
            print("🎆 MULTITALK SUCCESS!")
            print(f"\n✅ S3 Output: {result['s3_output']}")
            print(f"💾 Local: {result.get('local_output')}")
            print(f"📊 Size: {result['video_size']:,} bytes")
            print(f"🎬 Frames: {result['frame_count']}")
            print(f"🎵 Audio: {result['audio_duration']:.2f}s")
            print(f"🖼️ Image: {result['image_original']} -> {result['image_resized']}")
            print(f"🖥️ GPU: {result['gpu']}")
            print(f"🔧 Attention patched: {result.get('attention_patched', False)}")
            print("\n🎉 Your video has been successfully generated!")
            print("\nTo download from S3:")
            print(f"aws s3 cp {result['s3_output']} ./my_multitalk_video.mp4")
        else:
            print("❌ Failed")
            print(f"Error: {result.get('error', 'Unknown')[:500]}")
        print("="*60)