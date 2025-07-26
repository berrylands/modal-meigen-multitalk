#!/usr/bin/env python3
"""
MeiGen-MultiTalk with default 81 frames.
Try with standard settings first.
"""

import modal
import os

modal.enable_output()

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

# Minimal working image
minimal_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["git", "ffmpeg"])
    .pip_install(
        "torch==2.4.1",
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
        "numpy==1.24.4",
        "numba==0.59.1",
        "boto3",
        "huggingface_hub",
        "moviepy",
        "opencv-python",
        "scipy",
        "soundfile",
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
    )
    .env({"PYTHONPATH": "/root/MultiTalk"})
)

app = modal.App("multitalk-default")

model_volume = modal.Volume.from_name("multitalk-models", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("multitalk-hf-cache", create_if_missing=True)

@app.function(
    image=minimal_image,
    gpu="a10g",
    volumes={
        "/models": model_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("aws-secret"),
    ],
    timeout=900,
)
def test_default_inference():
    """
    Test with default 81 frames.
    """
    import boto3
    import tempfile
    import os
    import json
    import subprocess
    import sys
    from datetime import datetime
    
    print("="*60)
    print("MultiTalk Test with Default Settings")
    print("="*60)
    
    bucket_name = os.environ.get('AWS_BUCKET_NAME')
    if not bucket_name:
        return {"error": "AWS_BUCKET_NAME not found"}
    
    try:
        # Setup
        work_dir = tempfile.mkdtemp()
        s3 = boto3.client('s3')
        
        # Download files
        print("\nDownloading from S3...")
        image_path = os.path.join(work_dir, "input.png")
        audio_path = os.path.join(work_dir, "input.wav")
        
        s3.download_file(bucket_name, "multi1.png", image_path)
        s3.download_file(bucket_name, "1.wav", audio_path)
        print("✅ Downloaded")
        
        # Create simple test - just try to import and check
        print("\nChecking MultiTalk installation...")
        sys.path.insert(0, "/root/MultiTalk")
        
        # First just check if the script runs with help
        result = subprocess.run(
            ["python3", "/root/MultiTalk/generate_multitalk.py", "--help"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print("❌ generate_multitalk.py --help failed")
            print(f"Error: {result.stderr[:500]}")
            return {"success": False, "error": "Script won't run"}
        
        print("✅ Script runs")
        
        # Now try minimal generation with default settings
        input_data = {
            "prompt": "A person is speaking",
            "cond_image": image_path,
            "cond_audio": {"person1": audio_path}
        }
        
        input_json = os.path.join(work_dir, "input.json")
        with open(input_json, "w") as f:
            json.dump(input_data, f)
        
        output_path = os.path.join(work_dir, "output")
        
        # Minimal command - let MultiTalk use defaults
        print("\nRunning minimal inference...")
        cmd = [
            "python3", "/root/MultiTalk/generate_multitalk.py",
            "--ckpt_dir", "/models/base",
            "--wav2vec_dir", "/models/wav2vec",
            "--input_json", input_json,
            "--save_file", output_path,
            "--sample_steps", "10",  # Fewer steps for testing
        ]
        
        print(f"Command: {' '.join(cmd)}")
        
        # Run it
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=work_dir,
            timeout=300  # 5 minute timeout
        )
        
        print(f"\nReturn code: {result.returncode}")
        
        if result.returncode != 0:
            print("❌ Generation failed")
            print(f"\nSTDERR (last 1000 chars):\n{result.stderr[-1000:]}")
            print(f"\nSTDOUT (last 1000 chars):\n{result.stdout[-1000:]}")
            
            # Check if it's the frame mismatch issue
            if "shape" in result.stderr and "invalid" in result.stderr:
                print("\n⚠️  Shape mismatch detected")
                print("MultiTalk might need specific audio/video dimensions")
            
            return {
                "success": False,
                "error": "Generation failed",
                "stderr": result.stderr,
                "stdout": result.stdout
            }
        
        # Check output
        video_path = f"{output_path}.mp4"
        if os.path.exists(video_path):
            size = os.path.getsize(video_path)
            print(f"\n✅ Success! Video generated: {size:,} bytes")
            
            # Upload
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            s3_key = f"outputs/multitalk_test_{timestamp}.mp4"
            s3.upload_file(video_path, bucket_name, s3_key)
            
            return {
                "success": True,
                "s3_output": f"s3://{bucket_name}/{s3_key}",
                "size": size
            }
        else:
            print("❌ No output video found")
            print(f"Work dir contents: {os.listdir(work_dir)}")
            return {"success": False, "error": "No output"}
            
    except subprocess.TimeoutExpired:
        print("❌ Timeout!")
        return {"success": False, "error": "Timeout after 5 minutes"}
    except Exception as e:
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    with app.run():
        print("Testing MultiTalk with default settings...\n")
        
        result = test_default_inference.remote()
        
        print("\n" + "="*60)
        if result.get("success"):
            print("✅ TEST SUCCESSFUL!")
            print(f"Output: {result['s3_output']}")
            print(f"Size: {result['size']:,} bytes")
        else:
            print("❌ TEST FAILED!")
            print(f"Error: {result.get('error')}")
        print("="*60)
