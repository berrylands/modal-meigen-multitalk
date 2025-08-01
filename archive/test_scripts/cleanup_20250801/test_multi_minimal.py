#!/usr/bin/env python3
"""
Minimal multi-person test with maximum debugging.
"""

import modal
import os

app = modal.App("test-multi-minimal")

# Reuse the image and volumes from main app
from app_multitalk_cuda import multitalk_cuda_image, model_volume, hf_cache_volume

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
    timeout=600,  # 10 minutes
)
def test_minimal_multi():
    """Run minimal multi-person test with lots of debugging."""
    import json
    import subprocess
    import os
    import boto3
    
    os.chdir("/root/MultiTalk")
    
    # Download files
    bucket_name = os.environ.get('AWS_BUCKET_NAME')
    s3 = boto3.client('s3')
    
    print("Downloading test files...")
    s3.download_file(bucket_name, "multi1.png", "test.png")
    s3.download_file(bucket_name, "1.wav", "audio1.wav") 
    s3.download_file(bucket_name, "2.wav", "audio2.wav")
    
    # Create minimal JSON
    test_json = {
        "prompt": "Two people talking",
        "cond_image": "test.png",
        "audio_type": "para",
        "cond_audio": {
            "person1": "audio1.wav",
            "person2": "audio2.wav"
        }
    }
    
    with open("test.json", "w") as f:
        json.dump(test_json, f)
    
    print(f"\nTest JSON:")
    print(json.dumps(test_json, indent=2))
    
    # Run with MINIMAL parameters
    cmd = [
        "python3", "generate_multitalk.py",
        "--ckpt_dir", "weights/Wan2.1-I2V-14B-480P",
        "--wav2vec_dir", "weights/chinese-wav2vec2-base", 
        "--input_json", "test.json",
        "--frame_num", "21",  # Absolute minimum
        "--sample_steps", "2",  # Bare minimum
        "--num_persistent_param_in_dit", "8000000000",
        "--save_file", "test_output",
    ]
    
    print(f"\nRunning command:")
    print(" ".join(cmd))
    print("\n" + "="*60)
    
    # Run and stream output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    lines = []
    for line in iter(process.stdout.readline, ''):
        if line:
            print(line.rstrip())
            lines.append(line)
            if len(lines) > 100:  # Keep last 100 lines
                lines.pop(0)
    
    process.wait()
    print("="*60)
    print(f"Exit code: {process.returncode}")
    
    # Check output
    if os.path.exists("test_output.mp4"):
        size = os.path.getsize("test_output.mp4")
        print(f"\n✅ SUCCESS! Generated {size:,} bytes")
        
        # Upload to S3
        s3.upload_file("test_output.mp4", bucket_name, "outputs/test_multi_minimal.mp4")
        print(f"Uploaded to: s3://{bucket_name}/outputs/test_multi_minimal.mp4")
    else:
        print("\n❌ No output generated")
        
    return {
        "success": os.path.exists("test_output.mp4"),
        "exit_code": process.returncode,
        "last_lines": lines[-20:] if lines else []
    }


if __name__ == "__main__":
    result = test_minimal_multi.remote()
    print(f"\nResult: {result['success']}")
    if not result['success'] and result.get('last_lines'):
        print("\nLast 20 lines:")
        for line in result['last_lines']:
            print(line.rstrip())