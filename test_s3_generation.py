#!/usr/bin/env python3
"""
Test S3-enabled video generation with minimal image.
"""

import modal
import os

modal.enable_output()

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

# Use a minimal image for testing
test_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("boto3")
    .pip_install("torch", "torchvision")  # Basic ML deps for testing
)

app = modal.App("test-s3-generation")

@app.function(
    image=test_image,
    gpu="t4",  # Use T4 for faster startup
    timeout=300,
    secrets=[
        modal.Secret.from_name("aws-secret"),
        modal.Secret.from_name("huggingface-secret")
    ]
)
def test_generate_video():
    """Test video generation with S3 inputs."""
    import boto3
    import tempfile
    import os
    from datetime import datetime
    import shutil
    
    print("="*60)
    print("S3 Video Generation Test")
    print("="*60)
    
    # Get bucket name from Modal secret
    bucket_name = os.environ.get('AWS_BUCKET_NAME')
    if not bucket_name:
        return {"error": "AWS_BUCKET_NAME not found"}
    
    print(f"\nUsing bucket: {bucket_name}")
    
    try:
        # Initialize S3 client
        s3 = boto3.client('s3')
        
        # Create work directory
        work_dir = tempfile.mkdtemp(prefix="test_gen_")
        print(f"Working directory: {work_dir}")
        
        # Download inputs
        print("\nDownloading inputs...")
        
        # Download image
        image_path = os.path.join(work_dir, "input.png")
        s3.download_file(bucket_name, "multi1.png", image_path)
        print(f"✅ Downloaded multi1.png ({os.path.getsize(image_path):,} bytes)")
        
        # Download audio
        audio_path = os.path.join(work_dir, "input.wav")
        s3.download_file(bucket_name, "1.wav", audio_path)
        print(f"✅ Downloaded 1.wav ({os.path.getsize(audio_path):,} bytes)")
        
        # Test GPU availability
        print("\nChecking GPU...")
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  No GPU available")
        
        # Create test output
        print("\nCreating test output...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"test_output_{timestamp}.mp4"
        output_path = os.path.join(work_dir, output_name)
        
        # Create a simple test file
        with open(output_path, 'wb') as f:
            f.write(b"TEST VIDEO - S3 Integration Working!\n")
            f.write(f"Generated at: {timestamp}\n".encode())
            f.write(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}\n".encode())
        
        # Upload output
        print("\nUploading output...")
        s3_key = f"test_outputs/{output_name}"
        s3.upload_file(output_path, bucket_name, s3_key)
        
        s3_uri = f"s3://{bucket_name}/{s3_key}"
        print(f"✅ Uploaded to: {s3_uri}")
        
        # Clean up
        shutil.rmtree(work_dir)
        print("\n✅ Test completed successfully!")
        
        return {
            "success": True,
            "bucket": bucket_name,
            "output": s3_uri,
            "gpu_available": torch.cuda.is_available()
        }
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    with app.run():
        print("Testing S3-enabled generation...\n")
        
        result = test_generate_video.remote()
        
        print("\n" + "="*60)
        if result.get("success"):
            print("✅ TEST SUCCESSFUL!")
            print(f"Output: {result['output']}")
            print(f"GPU: {'Yes' if result['gpu_available'] else 'No'}")
        else:
            print("❌ TEST FAILED!")
            print(f"Error: {result.get('error')}")
        print("="*60)
