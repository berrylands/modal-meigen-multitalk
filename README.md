# Modal MeiGen-MultiTalk

A serverless implementation of [MeiGen-MultiTalk](https://github.com/MeiGen-AI/MultiTalk) running on [Modal](https://modal.com/) with Flash Attention 2.6.1 support.

## Overview

This project provides a serverless API for audio-driven multi-person conversational video generation using the MeiGen-MultiTalk model. It leverages Modal's serverless GPU infrastructure to provide on-demand video generation without maintaining expensive GPU instances.

## Features

- 🎥 Audio-driven talking head video generation
- 🚀 Serverless deployment with automatic scaling
- 💰 Pay-per-use pricing (no idle costs)
- 🎭 Support for single and multi-person conversations (up to multiple speakers)
- 🎨 Works with cartoon characters and singing
- ⚡ GPU-accelerated inference with Flash Attention 2.6.1
- ☁️ S3 integration for input/output storage
- 🔧 CUDA 12.1 optimized environment
- 🗣️ Label Rotary Position Embedding (L-RoPE) for correct speaker-audio binding

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

#### Single Person Video
Generate a single-person talking head video:

```bash
modal run app_multitalk_cuda.py
```

#### Two-Person Conversation
Generate a two-person conversation video:

```bash
modal run app_multitalk_cuda.py --two-person
```

Note: For two-person mode, ensure you have both `1.wav` and `2.wav` in your S3 bucket.

### Generate Video with Custom Parameters

Use the Modal functions directly for more control:

```python
# Single person
result = generate_video_cuda.remote(
    prompt="A person speaking about technology",
    image_key="portrait.png",
    audio_key="speech.wav",
    sample_steps=20
)

# Multiple people
result = generate_multi_person_video.remote(
    prompt="Two people having a conversation",
    image_key="two_people.png",
    audio_keys=["person1.wav", "person2.wav"],
    sample_steps=20
)
```

## Input Requirements

### Images
- **Resolution**: 896x448 pixels (exactly)
- **Format**: PNG or JPG
- **Content**: 
  - Single person: Portrait photo or character image
  - Multiple people: Image containing all speakers

### Audio
- **Format**: WAV file
- **Sample Rate**: Any (automatically resampled to 16kHz)
- **Duration**: Any length (frames calculated automatically)
- **Multi-person**: 
  - Provide separate audio files for each speaker
  - Audio files are synchronized to longest duration
  - Each speaker's audio is mapped to their position in the image

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

## Multi-Person Conversation Examples

### Two-Person Dialogue
```python
# Example: Interview scenario
result = generate_multi_person_video.remote(
    prompt="An interviewer and guest discussing artificial intelligence",
    image_key="interview_scene.png",  # Image with two people
    audio_keys=["interviewer.wav", "guest.wav"],
    sample_steps=20
)
```

### Three-Person Panel
```python
# Example: Panel discussion
result = generate_multi_person_video.remote(
    prompt="Three experts discussing climate change",
    image_key="panel_discussion.png",  # Image with three people
    audio_keys=["moderator.wav", "expert1.wav", "expert2.wav"],
    sample_steps=20
)
```

### JSON Input Format
The system automatically creates the correct JSON format for MultiTalk:
```json
{
  "prompt": "Two people having a conversation",
  "cond_image": "input.png",
  "cond_audio": {
    "person1": "input_person1.wav",
    "person2": "input_person2.wav"
  }
}
```

## Files

- `app_multitalk_cuda.py` - Main Modal application with Flash Attention and multi-person support
- `app_multitalk_multi_person.py` - Standalone multi-person implementation
- `s3_utils.py` - S3 integration utilities
- `modal_image.py` - Image definition configurations
- `MODAL_ML_LESSONS_LEARNED.md` - Detailed insights from the implementation journey

## License

This project follows the license terms of the original [MeiGen-MultiTalk](https://github.com/MeiGen-AI/MultiTalk) project.