#!/usr/bin/env python3
"""
Debug version of multi-person generation with real-time output.
"""

import modal
import os

app = modal.App("debug-multitalk")

# Use the same image from app_multitalk_cuda.py
multitalk_cuda_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04",
        add_python="3.10"
    )
    .apt_install([
        "git", "ffmpeg", "libsm6", "libxext6", 
        "libxrender-dev", "libgomp1", "wget",
    ])
    .pip_install(
        "torch==2.4.1", "torchvision==0.19.1", "torchaudio==2.4.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install("ninja", "packaging", "wheel", "setuptools")
    .pip_install("transformers==4.49.0")
    .pip_install("xformers==0.0.28", index_url="https://download.pytorch.org/whl/cu121")
    .run_commands("pip install flash-attn==2.6.1 --no-build-isolation")
    .pip_install([
        "peft", "accelerate", "diffusers>=0.30.0", "librosa", "moviepy",
        "opencv-python", "numpy==1.24.4", "numba==0.59.1", "scipy",
        "soundfile", "boto3", "huggingface_hub", "einops", "omegaconf",
        "tqdm", "optimum-quanto==0.2.6", "easydict", "ftfy", "pyloudnorm",
        "scikit-image", "Pillow", "misaki[en]",
    ])
    .run_commands(
        "cd /root && git clone https://github.com/MeiGen-AI/MultiTalk.git",
        "cd /root/MultiTalk && pip install -r requirements.txt || true",
    )
    .env({"PYTHONPATH": "/root/MultiTalk"})
)

@app.function(
    image=multitalk_cuda_image,
    gpu="a100-40gb",
    volumes={
        "/models": modal.Volume.from_name("multitalk-models"),
        "/root/.cache/huggingface": modal.Volume.from_name("multitalk-hf-cache"),
    },
    secrets=[
        modal.Secret.from_name("aws-secret"),
        modal.Secret.from_name("huggingface-secret"),
    ],
    timeout=1200,  # 20 minutes
)
def debug_multi_person():
    """
    Debug multi-person generation with detailed output.
    """
    import subprocess
    import sys
    import json
    import os
    
    print("Starting debug multi-person generation...")
    
    # Create a simple test JSON
    os.chdir("/root/MultiTalk")
    
    # Create test JSON manually
    test_json = {
        "prompt": "Two people having a conversation",
        "cond_image": "examples/single/1/out.png",  # Use example image
        "audio_type": "para",
        "cond_audio": {
            "person1": "examples/single/1/1.wav",  # Use example audio
            "person2": "examples/single/1/1.wav"   # Same audio for testing
        }
    }
    
    with open("test_multi.json", "w") as f:
        json.dump(test_json, f, indent=2)
    
    print("\nCreated test JSON:")
    print(json.dumps(test_json, indent=2))
    
    # Check if example files exist
    print("\nChecking example files:")
    for path in [test_json["cond_image"], test_json["cond_audio"]["person1"]]:
        exists = os.path.exists(path)
        print(f"  {path}: {'✅ exists' if exists else '❌ missing'}")
    
    # Run with real-time output
    print("\nRunning MultiTalk with streaming output...")
    print("-" * 60)
    
    cmd = [
        "python3", "generate_multitalk.py",
        "--ckpt_dir", "weights/Wan2.1-I2V-14B-480P",
        "--wav2vec_dir", "weights/chinese-wav2vec2-base",
        "--input_json", "test_multi.json",
        "--frame_num", "45",  # Smaller for testing
        "--sample_steps", "5",  # Very few steps for debugging
        "--num_persistent_param_in_dit", "8000000000",
        "--mode", "streaming",
        "--use_teacache",
        "--save_file", "debug_output",
    ]
    
    print("Command:", " ".join(cmd))
    print("-" * 60)
    
    # Run with real-time output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    )
    
    # Stream output
    for line in iter(process.stdout.readline, ''):
        if line:
            print(line.rstrip())
            sys.stdout.flush()
    
    process.wait()
    
    print("-" * 60)
    print(f"Process exited with code: {process.returncode}")
    
    # Check for output
    if os.path.exists("debug_output.mp4"):
        size = os.path.getsize("debug_output.mp4")
        print(f"\n✅ Output generated: debug_output.mp4 ({size:,} bytes)")
    else:
        print("\n❌ No output file generated")
        print("Files in directory:", os.listdir("."))
    
    return {"returncode": process.returncode}


if __name__ == "__main__":
    with app.run():
        result = debug_multi_person.remote()
        print(f"\nDebug completed with return code: {result.get('returncode')}")