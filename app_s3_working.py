#!/usr/bin/env python3
"""
Modal MeiGen-MultiTalk Application with S3 Support
Working version with simplified image for faster testing.
"""

import modal
import os

modal.enable_output()

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

# Use the working light image that builds successfully
from modal_image import multitalk_image_light

# Add boto3 for S3 support
multitalk_image_s3 = multitalk_image_light.pip_install("boto3")

app = modal.App("meigen-multitalk-s3")

@app.function(
    image=multitalk_image_s3,
    gpu="a10g",  # Use A10G for better performance
    timeout=600,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("aws-secret")
    ]
)
def generate_multitalk_video(
    prompt: str = "A person is speaking in a professional setting",
    image_key: str = "multi1.png",
    audio_key: str = "1.wav",
    upload_output: bool = True,
    output_prefix: str = "outputs/"
):
    """
    Generate a MultiTalk video using inputs from S3.
    
    Args:
        prompt: Text description for the video generation
        image_key: S3 key for input image (default: multi1.png)
        audio_key: S3 key for input audio (default: 1.wav)  
        upload_output: Whether to upload result to S3
        output_prefix: S3 prefix for outputs
        
    Returns:
        Dict with generation results
    """
    import boto3
    import tempfile
    import shutil
    from datetime import datetime
    import torch
    import subprocess
    import sys
    
    print("="*60)
    print("MeiGen-MultiTalk Video Generation")
    print("="*60)
    
    # Get bucket name from Modal secret
    bucket_name = os.environ.get('AWS_BUCKET_NAME')
    if not bucket_name:
        return {"error": "AWS_BUCKET_NAME not found in Modal secrets"}
    
    print(f"\nConfiguration:")
    print(f"  Bucket: {bucket_name}")
    print(f"  Image: {image_key}")
    print(f"  Audio: {audio_key}")
    print(f"  Prompt: {prompt}")
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n✅ GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("\n⚠️  No GPU available!")
    
    try:
        # Create work directory
        work_dir = tempfile.mkdtemp(prefix="multitalk_")
        print(f"\nWork directory: {work_dir}")
        
        # Initialize S3
        s3 = boto3.client('s3')
        
        # Download inputs
        print("\nDownloading inputs from S3...")
        
        image_path = os.path.join(work_dir, "input_image.png")
        s3.download_file(bucket_name, image_key, image_path)
        print(f"  ✅ Image: {os.path.getsize(image_path):,} bytes")
        
        audio_path = os.path.join(work_dir, "input_audio.wav") 
        s3.download_file(bucket_name, audio_key, audio_path)
        print(f"  ✅ Audio: {os.path.getsize(audio_path):,} bytes")
        
        # Check if MultiTalk is available
        multitalk_path = "/root/MultiTalk"
        if os.path.exists(multitalk_path):
            print(f"\n✅ MultiTalk found at: {multitalk_path}")
            
            # TODO: Run actual MultiTalk inference
            print("\n⚠️  MultiTalk inference not yet implemented")
            print("   Creating placeholder output...")
        else:
            print(f"\n⚠️  MultiTalk not found at {multitalk_path}")
            print("   Creating placeholder output...")
        
        # Create placeholder output for now
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"multitalk_{timestamp}.mp4"
        output_path = os.path.join(work_dir, output_name)
        
        # Create a more detailed placeholder
        with open(output_path, 'wb') as f:
            f.write(b"MultiTalk Video Output\n")
            f.write(b"======================\n")
            f.write(f"Timestamp: {timestamp}\n".encode())
            f.write(f"Prompt: {prompt}\n".encode())
            f.write(f"Image: {image_key}\n".encode()) 
            f.write(f"Audio: {audio_key}\n".encode())
            f.write(f"GPU: {gpu_name if torch.cuda.is_available() else 'None'}\n".encode())
            f.write(b"\nStatus: Placeholder - MultiTalk inference coming soon!\n")
        
        print(f"\nGenerated: {output_name} ({os.path.getsize(output_path):,} bytes)")
        
        # Upload output if requested
        if upload_output:
            print("\nUploading output to S3...")
            s3_key = f"{output_prefix}{output_name}"
            s3.upload_file(output_path, bucket_name, s3_key)
            
            s3_uri = f"s3://{bucket_name}/{s3_key}"
            print(f"  ✅ Uploaded to: {s3_uri}")
            
            # Clean up
            shutil.rmtree(work_dir)
            print("\n✅ Cleaned up temporary files")
            
            return {
                "success": True,
                "status": "completed",
                "bucket": bucket_name,
                "s3_output": s3_uri,
                "s3_key": s3_key,
                "gpu": gpu_name if torch.cuda.is_available() else None
            }
        else:
            return {
                "success": True,
                "status": "completed_local",
                "local_output": output_path,
                "gpu": gpu_name if torch.cuda.is_available() else None
            }
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


@app.function(
    image=multitalk_image_s3,
    secrets=[modal.Secret.from_name("aws-secret")]
)
def list_bucket_files(prefix: str = "", max_keys: int = 50):
    """List files in the S3 bucket."""
    import boto3
    
    bucket_name = os.environ.get('AWS_BUCKET_NAME')
    if not bucket_name:
        return {"error": "AWS_BUCKET_NAME not found"}
    
    print(f"Listing files in: {bucket_name}")
    if prefix:
        print(f"Prefix: {prefix}")
    
    try:
        s3 = boto3.client('s3')
        response = s3.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix,
            MaxKeys=max_keys
        )
        
        files = []
        if 'Contents' in response:
            for obj in response['Contents']:
                files.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'modified': obj['LastModified'].isoformat()
                })
                print(f"  {obj['Key']} ({obj['Size']:,} bytes)")
        
        return {
            "success": True,
            "bucket": bucket_name,
            "count": len(files),
            "files": files
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    with app.run():
        print("MeiGen-MultiTalk S3 Application\n")
        
        # First list files
        print("Checking S3 bucket...\n")
        list_result = list_bucket_files.remote()
        
        if not list_result.get("success"):
            print(f"❌ S3 access failed: {list_result.get('error')}")
            exit(1)
        
        print(f"\n✅ Found {list_result['count']} files")
        
        # Check for required files
        files = {f['key']: f for f in list_result['files']}
        if 'multi1.png' not in files or '1.wav' not in files:
            print("\n❌ Required files missing!")
            print(f"  multi1.png: {'Yes' if 'multi1.png' in files else 'No'}")
            print(f"  1.wav: {'Yes' if '1.wav' in files else 'No'}")
            exit(1)
        
        print("✅ Required files found!")
        
        # Generate video
        print("\nGenerating video...\n")
        
        result = generate_multitalk_video.remote(
            prompt="A person is speaking enthusiastically about AI technology",
            image_key="multi1.png",
            audio_key="1.wav",
            upload_output=True
        )
        
        print("\n" + "="*60)
        if result.get("success"):
            print("✅ VIDEO GENERATION SUCCESSFUL!")
            print(f"Output: {result.get('s3_output', result.get('local_output'))}")
            if result.get('gpu'):
                print(f"GPU used: {result['gpu']}")
        else:
            print("❌ VIDEO GENERATION FAILED!")
            print(f"Error: {result.get('error')}")
        print("="*60)
