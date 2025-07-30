#!/usr/bin/env python3
"""
Test multi-person generation with bounding boxes.
Based on official MultiTalk examples.
"""

import modal
import os

# Copy image definition from app_multitalk_cuda.py
multitalk_cuda_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04",
        add_python="3.10"
    )
    .apt_install([
        "git", "ffmpeg", "libsm6", "libxext6", 
        "libxrender-dev", "libgomp1", "wget",
    ])
    .pip_install("torch==2.4.1", "torchvision==0.19.1", 
                 "torchaudio==2.4.1", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("ninja", "packaging", "wheel", "setuptools")
    .pip_install("transformers==4.49.0")
    .pip_install("xformers==0.0.28", index_url="https://download.pytorch.org/whl/cu121")
    .run_commands("pip install flash-attn==2.6.1 --no-build-isolation")
    .pip_install(
        "peft", "accelerate", "diffusers>=0.30.0", "librosa", "moviepy",
        "opencv-python", "numpy==1.24.4", "numba==0.59.1", "scipy",
        "soundfile", "boto3", "huggingface_hub", "einops", "omegaconf",
        "tqdm", "optimum-quanto==0.2.6", "easydict", "ftfy", "pyloudnorm",
        "scikit-image", "Pillow", "misaki[en]",
    )
    .run_commands(
        "cd /root && git clone https://github.com/MeiGen-AI/MultiTalk.git",
        "cd /root/MultiTalk && pip install -r requirements.txt || true",
    )
    .env({"PYTHONPATH": "/root/MultiTalk"})
)

app = modal.App("test-bbox")
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
    timeout=1200,
)
def test_with_bounding_boxes():
    """Test multi-person with different configurations."""
    import json
    import subprocess
    import os
    import boto3
    from PIL import Image
    import shutil
    
    os.chdir("/root/MultiTalk")
    
    # Check Python executable
    import sys
    print(f"Python executable: {sys.executable}")
    
    # Download files
    bucket_name = os.environ.get('AWS_BUCKET_NAME')
    s3 = boto3.client('s3')
    
    print("Downloading test files...")
    s3.download_file(bucket_name, "multi1.png", "test_image.png")
    s3.download_file(bucket_name, "1.wav", "audio1.wav")
    s3.download_file(bucket_name, "2.wav", "audio2.wav")
    
    # Get image dimensions for bbox
    img = Image.open("test_image.png")
    width, height = img.size
    print(f"Image dimensions: {width}x{height}")
    
    # Test configurations based on official examples
    test_configs = [
        {
            "name": "para_no_bbox",
            "description": "Parallel mode without bounding boxes (like example 2)",
            "json": {
                "prompt": "Two people having an animated conversation",
                "cond_image": "test_image.png",
                "audio_type": "para",
                "cond_audio": {
                    "person1": "audio1.wav",
                    "person2": "audio2.wav"
                }
            }
        },
        {
            "name": "add_with_bbox",
            "description": "Additive mode with bounding boxes (like example 1)",
            "json": {
                "prompt": "Two people having an animated conversation",
                "cond_image": "test_image.png",
                "audio_type": "add",
                "cond_audio": {
                    "person1": "audio1.wav",
                    "person2": "audio2.wav"
                },
                "bbox": {
                    # Left half for person1, right half for person2
                    "person1": [0, 0, width//2, height],
                    "person2": [width//2, 0, width, height]
                }
            }
        },
        {
            "name": "para_with_bbox",
            "description": "Parallel mode with bounding boxes",
            "json": {
                "prompt": "Two people having an animated conversation",
                "cond_image": "test_image.png",
                "audio_type": "para",
                "cond_audio": {
                    "person1": "audio1.wav",
                    "person2": "audio2.wav"
                },
                "bbox": {
                    # Left half for person1, right half for person2
                    "person1": [0, 0, width//2, height],
                    "person2": [width//2, 0, width, height]
                }
            }
        }
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"JSON:")
        print(json.dumps(config['json'], indent=2))
        
        json_file = f"test_{config['name']}.json"
        with open(json_file, "w") as f:
            json.dump(config['json'], f, indent=2)
        
        # Copy MultiTalk weights
        base_dir = "weights/Wan2.1-I2V-14B-480P"
        multitalk_dir = "weights/MeiGen-MultiTalk"
        
        if os.path.exists(f"{base_dir}/diffusion_pytorch_model.safetensors.index.json"):
            shutil.move(
                f"{base_dir}/diffusion_pytorch_model.safetensors.index.json",
                f"{base_dir}/diffusion_pytorch_model.safetensors.index.json_backup"
            )
        
        files_to_copy = [
            "diffusion_pytorch_model.safetensors.index.json",
            "multitalk.safetensors"
        ]
        
        for f in files_to_copy:
            src = os.path.join(multitalk_dir, f)
            dst = os.path.join(base_dir, f)
            if os.path.exists(src):
                shutil.copy(src, dst)
        
        # Run test  
        cmd = [
            sys.executable, "generate_multitalk.py",
            "--ckpt_dir", base_dir,
            "--wav2vec_dir", "weights/chinese-wav2vec2-base",
            "--input_json", json_file,
            "--frame_num", "45",  # Quick test
            "--sample_steps", "5",  # Minimal steps
            "--num_persistent_param_in_dit", "11000000000",
            "--save_file", f"output_{config['name']}",
        ]
        
        print(f"\nRunning command:")
        print(" ".join(cmd))
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300,
                env={"CUDA_VISIBLE_DEVICES": "0"}
            )
            
            success = os.path.exists(f"output_{config['name']}.mp4")
            
            results.append({
                "config": config['name'],
                "success": success,
                "returncode": result.returncode,
                "error": result.stderr[-500:] if result.stderr else "No error"
            })
            
            if success:
                size = os.path.getsize(f"output_{config['name']}.mp4")
                print(f"\n✅ Success! Generated {size:,} bytes")
                
                # Upload to S3
                s3.upload_file(
                    f"output_{config['name']}.mp4",
                    bucket_name,
                    f"outputs/test_{config['name']}.mp4"
                )
                print(f"Uploaded to: s3://{bucket_name}/outputs/test_{config['name']}.mp4")
            else:
                print(f"\n❌ Failed")
                if result.stderr:
                    print(f"Error: {result.stderr[-500:]}")
                
        except subprocess.TimeoutExpired:
            results.append({
                "config": config['name'],
                "success": False,
                "returncode": -1,
                "error": "Timeout"
            })
            print("\n❌ Timeout")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY:")
    print(f"{'='*60}")
    
    for r in results:
        status = "✅" if r['success'] else "❌"
        print(f"{status} {r['config']}: {'Success' if r['success'] else 'Failed'}")
        if not r['success']:
            print(f"   Error: {r['error'][:100]}...")
    
    return results


if __name__ == "__main__":
    with app.run():
        results = test_with_bounding_boxes.remote()
        print("\nFinal results:")
        for r in results:
            print(f"  {r['config']}: {'✅' if r['success'] else '❌'}")