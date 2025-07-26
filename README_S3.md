# S3 Integration for Modal MeiGen-MultiTalk

## Overview

I've created S3 integration utilities that allow you to:
- Download input images and audio files from S3
- Upload generated videos back to S3
- Process files without manual download/upload

## Files Created

1. **s3_utils.py** - Core S3 utilities
   - `S3Manager` class for all S3 operations
   - Helper functions for easy integration

2. **app_s3.py** - S3-enabled Modal app
   - `generate_video_s3()` - Main function that downloads inputs, processes, and uploads output
   - `test_s3_access()` - Test function to verify S3 access

3. **test_s3_simple.py** - Simple S3 test script

## Usage

### 1. Set your S3 bucket name

```bash
export S3_BUCKET_NAME="your-bucket-name"
```

### 2. Test S3 access

```bash
python test_s3_simple.py your-bucket-name
```

### 3. Run video generation with S3

```bash
python app_s3.py your-bucket-name
```

This will:
- Download `multi1.png` and `1.wav` from your S3 bucket
- Process them (currently placeholder - actual inference not implemented)
- Upload the result back to S3 under `outputs/` prefix

## S3 File Structure

Expected input files in your bucket:
- `multi1.png` - Input image
- `1.wav` - Input audio

Output will be saved as:
- `outputs/multitalk_output_YYYYMMDD_HHMMSS.mp4`

## Integration with Main App

The S3 utilities can be integrated into any Modal function by:

1. Adding the mount:
```python
s3_utils_mount = modal.mount.from_local_file(
    local_path="s3_utils.py",
    remote_path="/root/s3_utils.py"
)
```

2. Using in your function:
```python
@app.function(
    mounts=[s3_utils_mount],
    secrets=[modal.Secret.from_name("aws-secret")]
)
def my_function():
    import sys
    sys.path.insert(0, '/root')
    from s3_utils import S3Manager
    
    s3 = S3Manager("my-bucket")
    # Use s3.download_file(), s3.upload_file(), etc.
```

## Required Secrets

Make sure you have AWS credentials configured in Modal:
- `aws-secret` with `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`

## Next Steps

Once we know your exact bucket name, we can:
1. Test the full S3 integration
2. Verify file downloads work correctly
3. Test upload functionality
4. Integrate with actual MultiTalk inference when ready