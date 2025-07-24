# Architecture Design

## Overview

Modal MeiGen-MultiTalk is a serverless API that uses the MeiGen-MultiTalk model to generate talking head videos from audio input. The system is designed to be called from AWS services with S3 integration for inputs and outputs.

## Core Components

### 1. MeiGen-MultiTalk Model Pipeline
- **Purpose**: Generate realistic talking head videos from audio
- **Models**:
  - MeiGen-MultiTalk (9.3 GB) - Main model for video generation
  - Wan2.1-I2V-14B (68.9 GB) - Base video generation model
  - Wav2Vec2 - Audio feature extraction
  - Additional models for enhancement
- **Key Features**:
  - Audio-driven facial animation
  - Lip-sync accuracy
  - Support for multiple speakers
  - Cartoon and realistic styles

### 2. Modal Serverless Infrastructure
- **GPU**: A10G (24GB) or A100 (40GB) for high-res
- **Storage**: Modal Volumes for 82GB+ model files
- **Scaling**: Auto-scales to zero when idle
- **Cold Start**: ~30-60s (model loading)
- **Inference Time**: 5-10s per video

### 3. S3 Integration Layer
```
Input Flow:
AWS Service → S3 Bucket → Modal Function → MeiGen Model → S3 Bucket → AWS Service

Options:
1. Direct S3 URLs as input
2. Pre-signed URLs for security
3. Batch processing from S3 prefix
```

### 4. REST API Design

```python
POST /generate-video
{
  "input": {
    "audio_s3_url": "s3://bucket/path/audio.wav",
    "reference_image_s3_url": "s3://bucket/path/face.jpg",  # Optional
    "prompt": "a person talking naturally"  # Optional
  },
  "output": {
    "s3_bucket": "output-bucket",
    "s3_prefix": "videos/",
    "filename": "output.mp4"  # Optional, auto-generated if not provided
  },
  "options": {
    "resolution": "480p",  # or "720p"
    "video_length": 5.0,   # seconds, max 15
    "turbo_mode": false,   # faster but lower quality
    "webhook_url": "https://..."  # Optional callback
  }
}

Response:
{
  "job_id": "job_123456",
  "status": "processing",
  "estimated_time": 10,
  "output_url": "s3://output-bucket/videos/output.mp4"
}
```

### 5. Processing Flow

1. **Request Reception**
   - Validate input parameters
   - Check S3 permissions
   - Create job ID

2. **Input Processing**
   - Download audio from S3
   - Validate audio format (convert to 16kHz if needed)
   - Download reference image (or use default)

3. **MeiGen-MultiTalk Generation**
   - Load models from Modal Volume
   - Extract audio features with Wav2Vec2
   - Generate motion with MultiTalk
   - Render video with WAN 2.1
   - Apply enhancements

4. **Output Delivery**
   - Encode video to MP4
   - Upload to S3 with metadata
   - Send webhook notification (if configured)
   - Clean up temporary files

## AWS Integration Patterns

### 1. Lambda Integration
```python
# AWS Lambda calling Modal
import requests
import boto3

def lambda_handler(event, context):
    response = requests.post(
        "https://modal-app-name.modal.run/generate-video",
        json={
            "input": {"audio_s3_url": event["audio_url"]},
            "output": {"s3_bucket": "my-output-bucket"}
        },
        headers={"Authorization": f"Bearer {MODAL_API_KEY}"}
    )
    return response.json()
```

### 2. Step Functions Integration
- Use for batch processing
- Handle retries and error states
- Chain with other AWS services

### 3. API Gateway Integration
- Proxy requests to Modal
- Add authentication/rate limiting
- Transform requests/responses

## Security Considerations

1. **S3 Access**
   - Use IAM roles for S3 access
   - Support pre-signed URLs
   - Validate S3 paths

2. **API Security**
   - Modal API key authentication
   - Optional JWT/OAuth2
   - Rate limiting

3. **Data Privacy**
   - Temporary file cleanup
   - No persistent storage of user data
   - Encrypted S3 transfers

## Performance Optimization

1. **Model Loading**
   - Pre-load models in container
   - Use Modal's warm pool
   - Optimize model formats

2. **S3 Transfers**
   - Stream large files
   - Use S3 Transfer Acceleration
   - Parallel uploads for batch

3. **GPU Utilization**
   - Batch multiple requests
   - Use appropriate GPU size
   - Enable mixed precision

## Monitoring & Observability

1. **Metrics**
   - Request count/latency
   - GPU utilization
   - S3 transfer times
   - Model inference times

2. **Logging**
   - Request/response logs
   - Error tracking
   - Performance profiling

3. **Alerts**
   - Failed jobs
   - High latency
   - S3 access errors