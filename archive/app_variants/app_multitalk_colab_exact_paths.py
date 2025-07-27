#!/usr/bin/env python3
"""
MeiGen-MultiTalk with EXACT Colab directory structure and paths.
This mimics how Colab runs from within the MultiTalk directory.
"""

import modal
import os

modal.enable_output()

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

# Colab-matching image
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
        # Clone MultiTalk to /content like Colab
        "mkdir -p /content",
        "cd /content && git clone https://github.com/MeiGen-AI/MultiTalk.git",
        "cd /content/MultiTalk && pip install -r requirements.txt || true",
    )
)

app = modal.App("multitalk-colab-exact-paths")

model_volume = modal.Volume.from_name("multitalk-models", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("multitalk-hf-cache", create_if_missing=True)

@app.function(
    image=colab_exact_image,
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
def generate_with_colab_paths(
    prompt: str = "A person is speaking enthusiastically about technology",
    image_key: str = "multi1.png",
    audio_key: str = "1.wav",
    sample_steps: int = 20
):
    """
    Generate video with EXACT Colab directory structure.
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
    from datetime import datetime
    from huggingface_hub import snapshot_download
    
    print("="*60)
    print("MeiGen-MultiTalk with Colab Directory Structure")
    print("="*60)
    
    bucket_name = os.environ.get('AWS_BUCKET_NAME')
    if not bucket_name:
        return {"error": "AWS_BUCKET_NAME not found"}
    
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"
    print(f"GPU: {gpu_name}")
    
    try:
        # CRITICAL: Set up Colab-style directory structure
        multitalk_dir = "/content/MultiTalk"
        weights_dir = os.path.join(multitalk_dir, "weights")
        
        # Create weights directory structure like Colab
        os.makedirs(weights_dir, exist_ok=True)
        
        print(f"\n📁 Setting up Colab directory structure...")
        print(f"  MultiTalk dir: {multitalk_dir}")
        print(f"  Weights dir: {weights_dir}")
        
        # Step 1: Download and set up models in Colab structure
        print("\n🤖 Setting up models in Colab structure...")
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        
        # Create symlinks from Modal volume to Colab structure
        model_mappings = [
            ("/models/base", os.path.join(weights_dir, "Wan2.1-I2V-14B-480P")),
            ("/models/wav2vec", os.path.join(weights_dir, "chinese-wav2vec2-base")),
            ("/models/multitalk", os.path.join(weights_dir, "MeiGen-MultiTalk")),
        ]
        
        for src, dst in model_mappings:
            if os.path.exists(src) and not os.path.exists(dst):
                print(f"  Linking {src} -> {dst}")
                os.symlink(src, dst)
            elif os.path.exists(dst):
                print(f"  ✅ {dst} already exists")
            else:
                print(f"  ❌ {src} not found, downloading...")
                # Download if missing
                repo_map = {
                    "Wan2.1-I2V-14B-480P": "Wan-AI/Wan2.1-I2V-14B-480P",
                    "chinese-wav2vec2-base": "TencentGameMate/chinese-wav2vec2-base",
                    "MeiGen-MultiTalk": "MeiGen-AI/MeiGen-MultiTalk",
                }
                model_name = os.path.basename(dst)
                if model_name in repo_map:
                    snapshot_download(
                        repo_id=repo_map[model_name],
                        local_dir=dst,
                        token=hf_token,
                        resume_download=True,
                    )
        
        # Step 2: Set up MultiTalk weights (Colab style)
        base_dir = os.path.join(weights_dir, "Wan2.1-I2V-14B-480P")
        multitalk_dir_weights = os.path.join(weights_dir, "MeiGen-MultiTalk")
        
        # Copy MultiTalk weights into base model (Colab commands)
        index_file = os.path.join(base_dir, "diffusion_pytorch_model.safetensors.index.json")
        if os.path.exists(index_file) and not os.path.exists(f"{index_file}_old"):
            shutil.move(index_file, f"{index_file}_old")
            print("  Backed up original index file")
        
        multitalk_files = [
            ("diffusion_pytorch_model.safetensors.index.json", index_file),
            ("multitalk.safetensors", os.path.join(base_dir, "multitalk.safetensors")),
        ]
        
        for src_file, dst_file in multitalk_files:
            src_path = os.path.join(multitalk_dir_weights, src_file)
            if os.path.exists(src_path) and not os.path.exists(dst_file):
                shutil.copy(src_path, dst_file)
                print(f"  Copied {src_file}")
        
        # Step 3: Change to MultiTalk directory (CRITICAL!)
        os.chdir(multitalk_dir)
        print(f"\n📂 Changed working directory to: {os.getcwd()}")
        
        # Step 4: Download inputs from S3 to current directory
        s3 = boto3.client('s3')
        
        print("\n⬇️ Downloading inputs from S3...")
        s3.download_file(bucket_name, image_key, "input.png")
        s3.download_file(bucket_name, audio_key, "input_raw.wav")
        print(f"  Downloaded to current directory")
        
        # Step 5: Audio preprocessing (resample to 16kHz)
        print("\n🎵 Audio preprocessing...")
        y, sr = librosa.load("input_raw.wav", sr=None)
        duration = len(y) / sr
        print(f"  Original: {duration:.2f}s @ {sr}Hz")
        
        # Resample to 16kHz
        y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sf.write("input.wav", y_16k, 16000, subtype='PCM_16')
        print(f"  Resampled to 16kHz")
        
        # Step 6: Create input JSON with relative paths
        input_data = {
            "prompt": prompt,
            "cond_image": "input.png",  # Relative path!
            "cond_audio": {
                "person1": "input.wav"  # Relative path!
            }
        }
        
        with open("input.json", "w") as f:
            json.dump(input_data, f, indent=2)
        
        print("\n📝 Input JSON (with relative paths):")
        print(json.dumps(input_data, indent=2))
        
        # Step 7: Run generate_multitalk.py from within MultiTalk directory
        print("\n🎬 Running MultiTalk (Colab-style)...")
        
        # VRAM parameter
        vram_param = 11000000000  # A100 default
        
        # EXACT Colab command with relative paths
        cmd = [
            "python3", "generate_multitalk.py",  # Relative path!
            "--ckpt_dir", "weights/Wan2.1-I2V-14B-480P",  # Relative path!
            "--wav2vec_dir", "weights/chinese-wav2vec2-base",  # Relative path!
            "--input_json", "input.json",
            "--sample_steps", str(sample_steps),
            "--num_persistent_param_in_dit", str(vram_param),
            "--mode", "streaming",
            "--use_teacache",
            "--save_file", "output",
        ]
        
        print(f"\nCommand (from {os.getcwd()}):")
        print(" ".join(cmd))
        print("\nNote: Running from within MultiTalk directory with relative paths!")
        
        # Add to Python path
        sys.path.insert(0, os.getcwd())
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print(f"\nReturn code: {result.returncode}")
        
        if result.returncode != 0:
            print(f"\n❌ Generation failed!")
            if result.stderr:
                print("\nSTDERR:")
                print(result.stderr[-2000:])
            if result.stdout:
                print("\nSTDOUT:")
                print(result.stdout[-1000:])
            
            return {
                "success": False,
                "error": "Generation failed",
                "stderr": result.stderr,
                "stdout": result.stdout,
                "working_dir": os.getcwd(),
                "colab_exact_paths": True
            }
        
        # Check for output
        if os.path.exists("output.mp4"):
            video_size = os.path.getsize("output.mp4")
            print(f"\n🎉 SUCCESS! Generated output.mp4 ({video_size:,} bytes)")
            
            # Upload to S3
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            s3_key = f"outputs/multitalk_colab_paths_{timestamp}.mp4"
            s3.upload_file("output.mp4", bucket_name, s3_key)
            
            s3_uri = f"s3://{bucket_name}/{s3_key}"
            print(f"✅ Uploaded to: {s3_uri}")
            
            # Copy locally
            local_copy = f"/tmp/multitalk_output_{timestamp}.mp4"
            shutil.copy("output.mp4", local_copy)
            
            return {
                "success": True,
                "s3_output": s3_uri,
                "local_output": local_copy,
                "video_size": video_size,
                "working_dir": os.getcwd(),
                "gpu": gpu_name,
                "colab_exact_paths": True
            }
        else:
            print("\n❌ No output.mp4 found")
            print("\nDirectory contents:")
            for f in os.listdir("."):
                print(f"  {f}")
            
            return {
                "success": False,
                "error": "No output found",
                "files": os.listdir(".")
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
        print("MeiGen-MultiTalk with EXACT Colab Paths\n")
        print("Key differences:")
        print("1. Uses /content/MultiTalk directory")
        print("2. Runs from within MultiTalk directory")
        print("3. Uses relative paths for everything")
        print("4. Models in weights/ subdirectory\n")
        
        result = generate_with_colab_paths.remote(
            prompt="A person is speaking enthusiastically about AI and technology",
            image_key="multi1.png",
            audio_key="1.wav",
            sample_steps=20
        )
        
        print("\n" + "="*60)
        if result.get("success"):
            print("🎆 SUCCESS WITH EXACT COLAB PATHS!")
            print(f"\nS3 Output: {result['s3_output']}")
            print(f"Local Output: {result.get('local_output')}")
            print(f"Video Size: {result['video_size']:,} bytes")
            print(f"Working Directory: {result['working_dir']}")
            print(f"GPU: {result['gpu']}")
            print("\n✅ This used EXACT Colab directory structure and paths!")
        else:
            print("❌ Generation Failed")
            print(f"Error: {result.get('error')}")
            print(f"Working Directory: {result.get('working_dir', 'Unknown')}")
        print("="*60)
