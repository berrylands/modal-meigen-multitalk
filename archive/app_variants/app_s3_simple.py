#!/usr/bin/env python3
"""
Simple S3-enabled Modal app for MultiTalk.
Uses basic image that builds quickly.
"""

import modal
import os

modal.enable_output()

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

# Simple image with just the essentials
simple_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("boto3", "torch", "torchvision", "transformers")
)

app = modal.App("multitalk-s3-simple")

@app.function(
    image=simple_image,
    gpu="t4",
    timeout=300,
    secrets=[
        modal.Secret.from_name("aws-secret"),
        modal.Secret.from_name("huggingface-secret")
    ]
)
def generate_video_s3():
    """
    Generate video using S3 inputs.
    Uses AWS_BUCKET_NAME from Modal secret.
    """
    import boto3
    import tempfile
    import os
    from datetime import datetime
    import torch
    
    print("="*60)
    print("MultiTalk S3 Video Generation (Simple)")
    print("="*60)
    
    # Get bucket
    bucket = os.environ.get('AWS_BUCKET_NAME')
    if not bucket:
        return {"error": "AWS_BUCKET_NAME not found"}
    
    print(f"\nBucket: {bucket}")
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        s3 = boto3.client('s3')
        work_dir = tempfile.mkdtemp()
        
        # Download
        print("\nDownloading...")
        image_path = os.path.join(work_dir, "input.png")
        audio_path = os.path.join(work_dir, "input.wav")
        
        s3.download_file(bucket, "multi1.png", image_path)
        s3.download_file(bucket, "1.wav", audio_path)
        print(f"✅ Downloaded files")
        
        # Process (placeholder)
        print("\nProcessing...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"multitalk_{timestamp}.mp4"
        output_path = os.path.join(work_dir, output_name)
        
        with open(output_path, 'wb') as f:
            f.write(f"MultiTalk output {timestamp}\n".encode())
            f.write(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}\n".encode())
        
        # Upload
        print("\nUploading...")
        s3_key = f"outputs/{output_name}"
        s3.upload_file(output_path, bucket, s3_key)
        
        result = f"s3://{bucket}/{s3_key}"
        print(f"✅ Uploaded to: {result}")
        
        return {
            "success": True,
            "output": result
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    with app.run():
        print("Running S3 video generation...\n")
        
        result = generate_video_s3.remote()
        
        print("\n" + "="*60)
        if result.get("success"):
            print("✅ SUCCESS!")
            print(f"Output: {result['output']}")
        else:
            print("❌ FAILED!")
            print(f"Error: {result.get('error')}")
        print("="*60)
