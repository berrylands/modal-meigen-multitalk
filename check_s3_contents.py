#!/usr/bin/env python3
"""
Check S3 bucket contents to see what test files are available.
"""

import modal
import os

app = modal.App("check-s3-contents")

simple_image = modal.Image.debian_slim().pip_install("boto3")

@app.function(
    image=simple_image,
    secrets=[modal.Secret.from_name("aws-secret")],
)
def list_s3_files():
    import boto3
    
    bucket_name = os.environ.get('AWS_BUCKET_NAME')
    s3 = boto3.client('s3')
    
    print(f"Checking S3 bucket: {bucket_name}\n")
    
    try:
        # List all objects
        response = s3.list_objects_v2(Bucket=bucket_name)
        
        if 'Contents' not in response:
            print("No files found in bucket")
            return []
        
        # Organize files by type
        images = []
        audio_files = []
        outputs = []
        other = []
        
        for obj in response['Contents']:
            key = obj['Key']
            size = obj['Size']
            
            if key.lower().endswith(('.png', '.jpg', '.jpeg')):
                images.append((key, size))
            elif key.lower().endswith(('.wav', '.mp3')):
                audio_files.append((key, size))
            elif key.startswith('outputs/'):
                outputs.append((key, size))
            else:
                other.append((key, size))
        
        # Print organized results
        print("📸 IMAGES:")
        for key, size in images:
            print(f"  - {key} ({size:,} bytes)")
        
        print(f"\n🎵 AUDIO FILES:")
        for key, size in audio_files:
            print(f"  - {key} ({size:,} bytes)")
        
        print(f"\n📹 OUTPUTS ({len(outputs)} files):")
        if outputs:
            # Show only recent outputs
            for key, size in outputs[-5:]:
                print(f"  - {key} ({size:,} bytes)")
            if len(outputs) > 5:
                print(f"  ... and {len(outputs) - 5} more")
        
        if other:
            print(f"\n📄 OTHER FILES:")
            for key, size in other:
                print(f"  - {key} ({size:,} bytes)")
        
        # Check for multi-person test data
        print("\n" + "="*50)
        print("🔍 MULTI-PERSON TEST DATA CHECK:")
        
        has_multi_image = any('multi' in k.lower() or 'two' in k.lower() or 'person' in k.lower() 
                             for k, _ in images)
        has_second_audio = any(k in [f[0] for f in audio_files] for k in ['2.wav', 'second.wav', 'person2.wav'])
        
        print(f"  Multi-person image available: {'✅' if has_multi_image else '❌'}")
        print(f"  Second audio file available: {'✅' if has_second_audio else '❌'}")
        
        if not has_second_audio:
            print("\n⚠️  Need to upload a second audio file (e.g., '2.wav') for two-person testing")
        
        return {
            'images': images,
            'audio_files': audio_files,
            'outputs': len(outputs),
            'has_multi_person_data': has_multi_image and has_second_audio
        }
        
    except Exception as e:
        print(f"Error listing bucket contents: {e}")
        return None


if __name__ == "__main__":
    with app.run():
        result = list_s3_files.remote()
        
        if result and not result.get('has_multi_person_data'):
            print("\n" + "="*50)
            print("📝 TO TEST MULTI-PERSON GENERATION:")
            print("1. Upload a second audio file as '2.wav'")
            print("2. Ensure 'multi1.png' contains two people")
            print("3. Run: modal run app_multitalk_cuda.py --two-person")