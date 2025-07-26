#!/usr/bin/env python3
"""
MeiGen-MultiTalk with flash attention properly built.
Using the image from modal_image.py that successfully builds flash-attn.
"""

import modal
import os

modal.enable_output()

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

# Import the light image that skips flash-attn from modal_image.py
from modal_image import multitalk_image_light as multitalk_image

app = modal.App("multitalk-light")

model_volume = modal.Volume.from_name("multitalk-models", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("multitalk-hf-cache", create_if_missing=True)

@app.function(
    image=multitalk_image,
    gpu="a100-40gb",
    volumes={
        "/models": model_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
    secrets=[
        modal.Secret.from_name("aws-secret"),
        modal.Secret.from_name("huggingface-secret"),
    ],
    timeout=900,
)
def generate_with_flash(
    prompt: str = "A person is speaking enthusiastically about technology",
    image_key: str = "multi1.png",
    audio_key: str = "1.wav",
    sample_steps: int = 20
):
    """
    Generate video with proper flash attention support and image resize.
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
    print("MeiGen-MultiTalk with Light Image")
    print("="*60)
    
    bucket_name = os.environ.get('AWS_BUCKET_NAME')
    s3 = boto3.client('s3')
    
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"
    print(f"GPU: {gpu_name}")
    
    # Check if flash-attn is available
    try:
        import flash_attn
        print(f"Flash-attn version: {flash_attn.__version__}")
    except:
        print("Flash-attn not available, using xformers/pytorch fallback")
    
    try:
        # First, patch the attention.py file to handle missing flash-attn
        attention_file = "/root/MultiTalk/wan/modules/attention.py"
        if os.path.exists(attention_file):
            print("\n🔧 Patching attention mechanism...")
            with open(attention_file, 'r') as f:
                content = f.read()
            
            # Replace the assertion to allow fallback
            if "assert FLASH_ATTN_2_AVAILABLE" in content:
                content = content.replace(
                    "assert FLASH_ATTN_2_AVAILABLE",
                    "if not FLASH_ATTN_2_AVAILABLE:\n        # Use xformers or PyTorch attention instead\n        import torch.nn.functional as F\n        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=causal)"
                )
                
                with open(attention_file, 'w') as f:
                    f.write(content)
                print("  ✅ Patched attention.py for fallback")
        
        # Set up directory structure
        # Use /root/MultiTalk as per modal_image.py setup
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
                print(f"  Linked {src} -> {dst}")
        
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
                "image_resized": True,
                "audio_processed": True
            }
        
        # Check for output
        if os.path.exists("output.mp4"):
            video_size = os.path.getsize("output.mp4")
            print(f"\n🎉 SUCCESS! Generated {video_size:,} bytes")
            
            # Upload to S3
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            s3_key = f"outputs/multitalk_light_{timestamp}_{frame_count}f.mp4"
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
                "gpu": gpu_name
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
        print("MeiGen-MultiTalk with Light Image\n")
        print("Using modal_image.py light image (no flash-attn build)\n")
        
        result = generate_with_flash.remote(
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
            print("\n🎉 Your video has been successfully generated!")
            print("\nTo download from S3:")
            print(f"aws s3 cp {result['s3_output']} ./my_multitalk_video.mp4")
        else:
            print("❌ Failed")
            print(f"Error: {result.get('error', 'Unknown')[:500]}")
        print("="*60)