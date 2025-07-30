#!/usr/bin/env python3
"""
Detailed debug of multi-person generation.
"""

import modal
import os
from app_multitalk_cuda import app, multitalk_cuda_image, model_volume, hf_cache_volume

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
    timeout=1800,  # 30 minutes
)
def debug_multi_person_detailed():
    """
    Debug multi-person with detailed logging and checks.
    """
    import boto3
    import json
    import subprocess
    import sys
    import os
    import shutil
    import torch
    
    print("="*60)
    print("DETAILED MULTI-PERSON DEBUG")
    print("="*60)
    
    # Check GPU
    print(f"\n1. GPU Check:")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Check models
    print(f"\n2. Model Check:")
    model_paths = [
        "/models/base",
        "/models/wav2vec", 
        "/models/multitalk"
    ]
    for path in model_paths:
        exists = os.path.exists(path)
        print(f"   {path}: {'✅' if exists else '❌'}")
    
    # Create minimal test
    os.chdir("/root/MultiTalk")
    
    # Download test files from S3
    bucket_name = os.environ.get('AWS_BUCKET_NAME')
    s3 = boto3.client('s3')
    
    print(f"\n3. Downloading test files from S3...")
    s3.download_file(bucket_name, "multi1.png", "test_image.png")
    s3.download_file(bucket_name, "1.wav", "test_audio1.wav")
    s3.download_file(bucket_name, "2.wav", "test_audio2.wav")
    print("   ✅ Downloaded all files")
    
    # Check downloaded files
    import librosa
    from PIL import Image
    
    print(f"\n4. File Analysis:")
    img = Image.open("test_image.png")
    print(f"   Image: {img.size} {img.mode}")
    
    for i, audio_file in enumerate(["test_audio1.wav", "test_audio2.wav"]):
        y, sr = librosa.load(audio_file, sr=None)
        print(f"   Audio {i+1}: {len(y)/sr:.2f}s @ {sr}Hz")
    
    # Create test JSON
    test_json = {
        "prompt": "Two people having a conversation",
        "cond_image": "test_image.png",
        "audio_type": "para",
        "cond_audio": {
            "person1": "test_audio1.wav",
            "person2": "test_audio2.wav"
        }
    }
    
    with open("debug_multi.json", "w") as f:
        json.dump(test_json, f, indent=2)
    
    print(f"\n5. Test JSON:")
    print(json.dumps(test_json, indent=2))
    
    # Run with minimal parameters first
    print(f"\n6. Running MultiTalk (minimal test)...")
    print("-" * 60)
    
    cmd = [
        "python3", "generate_multitalk.py",
        "--ckpt_dir", "weights/Wan2.1-I2V-14B-480P",
        "--wav2vec_dir", "weights/chinese-wav2vec2-base",
        "--input_json", "debug_multi.json",
        "--frame_num", "21",  # Minimum frames
        "--sample_steps", "2",  # Minimum steps
        "--num_persistent_param_in_dit", "8000000000",
        "--save_file", "debug_output",
    ]
    
    # First try without streaming or teacache
    print("Command:", " ".join(cmd))
    print("-" * 60)
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    # Run with output capture
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        timeout=300  # 5 minute timeout for minimal test
    )
    
    print(f"\nReturn code: {result.returncode}")
    
    if result.stdout:
        print("\nSTDOUT:")
        print(result.stdout[-2000:])
    
    if result.stderr:
        print("\nSTDERR:")
        print(result.stderr[-2000:])
    
    # Check for output
    output_exists = os.path.exists("debug_output.mp4")
    print(f"\n7. Output check:")
    print(f"   debug_output.mp4 exists: {'✅' if output_exists else '❌'}")
    
    if output_exists:
        size = os.path.getsize("debug_output.mp4")
        print(f"   Size: {size:,} bytes")
    
    # List files in directory
    print(f"\n8. Files in directory:")
    for f in sorted(os.listdir(".")):
        if any(x in f for x in ['debug', 'test', '.mp4', '.json', '.wav', '.png']):
            print(f"   {f}")
    
    return {
        "success": output_exists,
        "return_code": result.returncode,
        "has_stderr": bool(result.stderr)
    }


if __name__ == "__main__":
    with app.run():
        print("Starting detailed multi-person debug...")
        result = debug_multi_person_detailed.remote()
        
        print("\n" + "="*60)
        print("DEBUG RESULTS:")
        print(f"  Success: {result.get('success')}")
        print(f"  Return code: {result.get('return_code')}")
        print(f"  Has errors: {result.get('has_stderr')}")
        print("="*60)