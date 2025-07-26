"""
Modal MeiGen-MultiTalk Application with S3 Support (Final Version)
"""

import modal
from modal import App
import os
from typing import Optional, Dict, Any

# Import our image definition  
from modal_image import multitalk_image_light as multitalk_image

# Define the Modal app
app = App("meigen-multitalk-s3-final")

# Include S3 utilities directly in the image
multitalk_image_with_s3 = multitalk_image.pip_install("boto3")

@app.function(
    image=multitalk_image_with_s3,
    gpu="a10g",
    timeout=600,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("aws-secret")
    ]
)
def generate_video_s3(
    prompt: str = "A person is speaking in a professional setting",
    image_key: str = "multi1.png",
    audio_key: str = "1.wav",
    upload_output: bool = True,
    output_prefix: str = "outputs/"
) -> Dict[str, Any]:
    """
    Generate a video using inputs from S3 and optionally upload the result.
    
    Uses AWS_BUCKET_NAME from Modal secret automatically.
    """
    import boto3
    import tempfile
    import os
    from datetime import datetime
    import shutil
    
    print("="*60)
    print("MeiGen-MultiTalk S3 Video Generation")
    print("="*60)
    
    # Get bucket name from Modal secret
    bucket_name = os.environ.get('AWS_BUCKET_NAME')
    if not bucket_name:
        return {"status": "error", "error": "AWS_BUCKET_NAME not found in Modal secrets"}
    
    print(f"\nUsing S3 bucket: {bucket_name}")
    
    results = {
        "status": "started",
        "prompt": prompt,
        "bucket": bucket_name,
        "image_key": image_key,
        "audio_key": audio_key,
    }
    
    try:
        # Initialize S3 client
        s3 = boto3.client('s3')
        
        # Create temporary directory
        work_dir = tempfile.mkdtemp(prefix="multitalk_")
        print(f"\nWorking directory: {work_dir}")
        
        # Download inputs
        print(f"\nDownloading inputs from S3...")
        
        # Download image
        image_path = os.path.join(work_dir, "input_image.png")
        print(f"  Downloading {image_key}...")
        s3.download_file(bucket_name, image_key, image_path)
        image_size = os.path.getsize(image_path)
        print(f"  ✅ Downloaded image ({image_size:,} bytes)")
        
        # Download audio
        audio_path = os.path.join(work_dir, "input_audio.wav")
        print(f"  Downloading {audio_key}...")
        s3.download_file(bucket_name, audio_key, audio_path)
        audio_size = os.path.getsize(audio_path)
        print(f"  ✅ Downloaded audio ({audio_size:,} bytes)")
        
        results["image_size"] = image_size
        results["audio_size"] = audio_size
        
        # TODO: Actual MultiTalk inference here
        print(f"\nRunning MultiTalk inference...")
        print(f"  Prompt: {prompt}")
        print(f"  ⚠️  NOTE: Actual inference not yet implemented")
        print(f"  Creating placeholder output...")
        
        # For now, create a placeholder output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"multitalk_output_{timestamp}.mp4"
        output_path = os.path.join(work_dir, output_filename)
        
        # Create placeholder file
        with open(output_path, 'wb') as f:
            f.write(b"PLACEHOLDER VIDEO - MultiTalk inference coming soon!\n")
            f.write(f"Prompt: {prompt}\n".encode())
            f.write(f"Image: {image_key} ({image_size} bytes)\n".encode())
            f.write(f"Audio: {audio_key} ({audio_size} bytes)\n".encode())
        
        output_size = os.path.getsize(output_path)
        results["output_size"] = output_size
        
        # Upload output if requested
        if upload_output:
            print(f"\nUploading output to S3...")
            s3_key = f"{output_prefix}{output_filename}"
            s3.upload_file(output_path, bucket_name, s3_key)
            
            s3_uri = f"s3://{bucket_name}/{s3_key}"
            print(f"  ✅ Uploaded to: {s3_uri}")
            
            results["s3_output"] = s3_uri
            results["s3_key"] = s3_key
            results["status"] = "completed"
        else:
            results["status"] = "completed_local_only"
            results["local_output"] = output_path
        
        # Clean up
        print(f"\nCleaning up temporary files...")
        shutil.rmtree(work_dir)
        print("  ✅ Cleaned up")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        results["status"] = "failed"
        results["error"] = str(e)
    
    print("\n" + "="*60)
    print(f"Status: {results['status'].upper()}")
    if results.get('s3_output'):
        print(f"Output: {results['s3_output']}")
    print("="*60)
    
    return results


@app.function(
    image=multitalk_image_with_s3,
    secrets=[modal.Secret.from_name("aws-secret")],
)
def list_s3_files() -> Dict[str, Any]:
    """List files in the S3 bucket."""
    import boto3
    import os
    
    bucket_name = os.environ.get('AWS_BUCKET_NAME')
    if not bucket_name:
        return {"success": False, "error": "AWS_BUCKET_NAME not found"}
    
    print(f"Listing files in bucket: {bucket_name}\n")
    
    try:
        s3 = boto3.client('s3')
        response = s3.list_objects_v2(Bucket=bucket_name, MaxKeys=50)
        
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
    print("MeiGen-MultiTalk S3 Video Generation\n")
    
    with app.run():
        # First list files to verify access
        print("Checking S3 access...\n")
        list_result = list_s3_files.remote()
        
        if not list_result["success"]:
            print(f"❌ S3 access failed: {list_result.get('error')}")
            exit(1)
    
        print(f"\n✅ S3 access successful!")
        print(f"Found {list_result['count']} files in bucket\n")
        
        # Check for required files
        files = {f['key']: f for f in list_result['files']}
        if 'multi1.png' not in files or '1.wav' not in files:
            print("❌ Required files not found!")
            print(f"  multi1.png: {'✅' if 'multi1.png' in files else '❌'}")
            print(f"  1.wav: {'✅' if '1.wav' in files else '❌'}")
            exit(1)
        
        print("✅ Required files found!\n")
        
        # Run video generation
        print("Starting video generation...\n")
        
        result = generate_video_s3.remote(
            prompt="A person is speaking in a professional studio setting",
            image_key="multi1.png",
            audio_key="1.wav",
            upload_output=True
        )
        
        if result["status"] == "completed":
            print(f"\n✅ Success!")
            print(f"Output uploaded to: {result['s3_output']}")
            print(f"\nYou can download it with:")
            print(f"  aws s3 cp {result['s3_output']} ./output.mp4")
        else:
            print(f"\n❌ Failed: {result.get('error', 'Unknown error')}")