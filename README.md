# Modal MeiGen-MultiTalk

A serverless implementation of [MeiGen-MultiTalk](https://github.com/MeiGen-AI/MultiTalk) running on [Modal](https://modal.com/) with Flash Attention 2.6.1 support.

## Overview

This project provides a serverless API for audio-driven multi-person conversational video generation using the MeiGen-MultiTalk model. It leverages Modal's serverless GPU infrastructure to provide on-demand video generation without maintaining expensive GPU instances.

## Features

- 🎥 Audio-driven talking head video generation
- 🚀 Serverless deployment with automatic scaling
- 💰 Pay-per-use pricing (no idle costs)
- 🎭 Support for single and multi-person scenarios
- 🎨 Works with cartoon characters and singing
- ⚡ GPU-accelerated inference with Flash Attention 2.6.1
- ☁️ S3 integration for input/output storage
- 🔧 CUDA 12.1 optimized environment

## Requirements

- Python 3.10+
- Modal account and API key
- AWS credentials (for S3 integration)
- ~82GB storage for model weights

## Setup

### 1. Install Dependencies

```bash
pip install modal boto3
```

### 2. Configure Modal

```bash
modal setup
```

### 3. Configure Secrets

Create the following secrets in your Modal dashboard:

#### AWS Secret (name: `aws-secret`)
```json
{
  "AWS_ACCESS_KEY_ID": "your-access-key-id",
  "AWS_SECRET_ACCESS_KEY": "your-secret-access-key",
  "AWS_BUCKET_NAME": "your-bucket-name"
}
```

#### Hugging Face Secret (name: `huggingface-secret`)
```json
{
  "HF_TOKEN": "your-huggingface-token"
}
```

### 4. Download Models

Before first use, download the required models:

```bash
modal run app_multitalk_cuda.py --action download
```

This downloads:
- Wan2.1-I2V-14B-480P (base model)
- chinese-wav2vec2-base (audio encoder)
- MeiGen-MultiTalk weights

## Usage

### Test Environment

Verify your setup:

```bash
modal run app_multitalk_cuda.py --action test
```

### Generate Video with S3

Generate a video using files from S3:

```bash
modal run app_multitalk_cuda.py --action generate-s3 \
  --bucket your-bucket-name \
  --image-key path/to/image.png \
  --audio-key path/to/audio.wav
```

### Generate Video with Local Files

Generate a video using local files:

```bash
modal run app_multitalk_cuda.py --action generate \
  --image-path /path/to/image.png \
  --audio-path /path/to/audio.wav \
  --prompt "A person is speaking"
```

## Input Requirements

### Images
- **Resolution**: 896x448 pixels (exactly)
- **Format**: PNG or JPG
- **Content**: Portrait photo or character image

### Audio
- **Format**: WAV file
- **Sample Rate**: Any (automatically resampled to 16kHz)
- **Duration**: Any length (frames calculated automatically)

## Output

- **Format**: MP4 video
- **Resolution**: 896x448 pixels
- **Frame Rate**: 24 fps
- **Location**: Uploaded to S3 bucket under `outputs/` prefix

## Advanced Configuration

### GPU Selection

The system uses A10G GPUs by default. To use different GPUs:

```python
# In app_multitalk_cuda.py
@app.function(gpu="a100")  # Options: t4, a10g, a100
```

### VRAM Parameters

Adjust based on GPU memory:
- A10G (24GB): 8000000000
- A100 (40GB): 11000000000
- A100 (80GB): 22000000000

### Sample Steps

Control generation quality/speed:
```bash
--sample-steps 20  # Default: 20, Range: 10-50
```

## Architecture

The implementation uses:
- **Base Image**: NVIDIA CUDA 12.1 development environment
- **PyTorch**: 2.4.1 with CUDA 12.1 support
- **Flash Attention**: 2.6.1 for efficient attention computation
- **xformers**: 0.0.28 for memory-efficient transformers
- **Modal Volumes**: Persistent storage for 82GB of model weights

## Troubleshooting

### Flash Attention Issues

If you encounter Flash Attention errors:
1. Ensure you're using the CUDA base image version
2. Check GPU compatibility (requires compute capability ≥ 7.5)

### Image Dimension Errors

Always resize images to exactly 896x448 pixels. The model architecture requires this specific resolution.

### Audio Length Errors

The system automatically calculates the correct frame count based on audio duration. Frame counts must follow the pattern 4n+1 (e.g., 21, 45, 81, 121, 161, 201).

### Memory Issues

If you encounter OOM errors:
1. Reduce sample steps
2. Use a larger GPU (A100 recommended)
3. Adjust VRAM parameters

## Development

For development setup and contributing guidelines, see [DEVELOPMENT.md](DEVELOPMENT.md).

## Files

- `app_multitalk_cuda.py` - Main Modal application with Flash Attention support
- `s3_utils.py` - S3 integration utilities
- `modal_image.py` - Image definition configurations
- `debug_*.py` - Debugging and testing utilities

## License

This project follows the license terms of the original [MeiGen-MultiTalk](https://github.com/MeiGen-AI/MultiTalk) project.