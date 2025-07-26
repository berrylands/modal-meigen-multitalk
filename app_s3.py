"""
Modal MeiGen-MultiTalk Application with S3 Support
"""

import modal
from modal import App, method
import os
from typing import Optional, Dict, Any

# Import our image definition
# Note: This is a simplified image for testing - production image needs full testing
from modal_image import multitalk_image_light as multitalk_image

# Define the Modal app
app = App("meigen-multitalk-s3")

# Mount S3 utilities
s3_utils_mount = modal.mount.from_local_file(
    local_path="s3_utils.py",
    remote_path="/root/s3_utils.py"
)

@app.function(
    image=multitalk_image,
    gpu="a10g",
    mounts=[s3_utils_mount],
    timeout=600,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("aws-secret")
    ]
)
def generate_video_s3(
    prompt: str,
    s3_bucket: str = None,  # Optional - will use AWS_BUCKET_NAME from secret if not provided
    image_key: str = "multi1.png",
    audio_key: str = "1.wav",
    upload_output: bool = True,
    output_prefix: str = "outputs/"
) -> Dict[str, Any]:
    """
    Generate a video using inputs from S3 and optionally upload the result.
    
    Args:
        prompt: Text description for the video
        s3_bucket: S3 bucket name
        image_key: S3 key for input image (default: multi1.png)
        audio_key: S3 key for input audio (default: 1.wav)
        upload_output: Whether to upload the result to S3
        output_prefix: S3 prefix for output files
        
    Returns:
        Dict with results including output location
    """
    import sys
    import tempfile
    import os
    from datetime import datetime
    
    sys.path.insert(0, '/root')
    from s3_utils import S3Manager
    
    print("="*60)
    print("MeiGen-MultiTalk S3 Video Generation")
    print("="*60)
    
    results = {
        "status": "started",
        "prompt": prompt,
        "bucket": s3_bucket,
        "image_key": image_key,
        "audio_key": audio_key,
        "errors": []
    }
    
    try:
        # Initialize S3 manager (will use AWS_BUCKET_NAME from secret if bucket not provided)
        if not s3_bucket:
            s3_bucket = os.environ.get('AWS_BUCKET_NAME')
            print(f"\n1. Using bucket from Modal secret: {s3_bucket}")
        else:
            print(f"\n1. Using provided bucket: {s3_bucket}")
        
        s3_manager = S3Manager(s3_bucket)
        
        # Download inputs
        print(f"\n2. Downloading inputs from S3...")
        print(f"   Image: s3://{s3_bucket}/{image_key}")
        print(f"   Audio: s3://{s3_bucket}/{audio_key}")
        
        inputs = s3_manager.download_inputs(image_key, audio_key)
        
        print(f"   ✅ Downloaded to: {inputs['output_dir']}")
        print(f"   Image: {os.path.basename(inputs['image_path'])} ({os.path.getsize(inputs['image_path']):,} bytes)")
        print(f"   Audio: {os.path.basename(inputs['audio_path'])} ({os.path.getsize(inputs['audio_path']):,} bytes)")
        
        results["local_image"] = inputs['image_path']
        results["local_audio"] = inputs['audio_path']
        
        # TODO: Actual MultiTalk inference here
        print(f"\n3. Running MultiTalk inference...")
        print(f"   Prompt: {prompt}")
        print(f"   ⚠️  NOTE: Actual inference not yet implemented")
        print(f"   Creating placeholder output...")
        
        # For now, create a placeholder output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"multitalk_output_{timestamp}.mp4"
        output_path = os.path.join(inputs['output_dir'], output_filename)
        
        # Create placeholder file
        with open(output_path, 'wb') as f:
            f.write(b"PLACEHOLDER VIDEO CONTENT - MultiTalk inference not yet implemented")
        
        results["local_output"] = output_path
        results["output_size"] = os.path.getsize(output_path)
        
        # Upload output if requested
        if upload_output:
            print(f"\n4. Uploading output to S3...")
            s3_key = f"{output_prefix}{output_filename}"
            s3_uri = s3_manager.upload_file(output_path, s3_key)
            print(f"   ✅ Uploaded to: {s3_uri}")
            
            results["s3_output"] = s3_uri
            results["status"] = "completed"
        else:
            results["status"] = "completed_local_only"
        
        # Clean up temporary files
        print(f"\n5. Cleaning up temporary files...")
        import shutil
        shutil.rmtree(inputs['output_dir'])
        print("   ✅ Cleaned up")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        results["status"] = "failed"
        results["errors"].append(str(e))
    
    print("\n" + "="*60)
    print(f"Status: {results['status'].upper()}")
    if results.get('s3_output'):
        print(f"Output: {results['s3_output']}")
    print("="*60)
    
    return results


@app.function(
    image=multitalk_image,
    secrets=[modal.Secret.from_name("aws-secret")],
    mounts=[s3_utils_mount],
)
def test_s3_access(bucket_name: str = None) -> Dict[str, Any]:
    """Test S3 access and list available files."""
    import sys
    sys.path.insert(0, '/root')
    from s3_utils import S3Manager
    
    if not bucket_name:
        bucket_name = os.environ.get('AWS_BUCKET_NAME')
        print(f"Using bucket from Modal secret: {bucket_name}")
    else:
        print(f"Testing S3 access to bucket: {bucket_name}")
    
    try:
        s3_manager = S3Manager(bucket_name)
        contents = s3_manager.list_bucket_contents()
        
        print(f"\nFound {len(contents)} objects in bucket:")
        for obj in contents:
            print(f"  - {obj}")
        
        # Check for our expected files
        has_image = "multi1.png" in contents
        has_audio = "1.wav" in contents
        
        return {
            "success": True,
            "bucket": bucket_name,
            "total_objects": len(contents),
            "objects": contents,
            "has_multi1_png": has_image,
            "has_1_wav": has_audio,
            "ready": has_image and has_audio
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.local_entrypoint()
def main():
    """Run S3-enabled video generation."""
    import sys
    
    # Get bucket name from argument (optional - will use Modal secret if not provided)
    bucket_name = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Test S3 access first
    if bucket_name:
        print(f"Testing S3 access to bucket: {bucket_name}\n")
    else:
        print("Testing S3 access using bucket from Modal secret...\n")
    
    test_result = test_s3_access.remote(bucket_name)
    
    if not test_result["success"]:
        print(f"❌ S3 access failed: {test_result.get('error')}")
        return
    
    if not test_result["ready"]:
        print("❌ Required files not found in bucket:")
        print(f"  multi1.png: {'✅' if test_result['has_multi1_png'] else '❌'}")
        print(f"  1.wav: {'✅' if test_result['has_1_wav'] else '❌'}")
        return
    
    print("✅ S3 access successful, files found!\n")
    
    # Run generation
    print("Running video generation with S3 inputs...\n")
    
    result = generate_video_s3.remote(
        prompt="A person is speaking in a professional setting",
        s3_bucket=bucket_name,
        image_key="multi1.png",
        audio_key="1.wav",
        upload_output=True
    )
    
    if result["status"] == "completed":
        print(f"\n✅ Success! Output uploaded to: {result['s3_output']}")
    else:
        print(f"\n❌ Failed: {result.get('errors')}")


if __name__ == "__main__":
    main()