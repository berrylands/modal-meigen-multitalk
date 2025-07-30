# Modal MeiGen-MultiTalk

A serverless implementation of [MeiGen-MultiTalk](https://github.com/MeiGen-AI/MultiTalk) running on [Modal](https://modal.com/) with Flash Attention 2.6.1 support.

## Overview

This project provides a serverless API for audio-driven multi-person conversational video generation using the MeiGen-MultiTalk model. It leverages Modal's serverless GPU infrastructure to provide on-demand video generation without maintaining expensive GPU instances.

## Features

- 🎥 Audio-driven talking head video generation
- 🚀 Serverless deployment with automatic scaling
- 💰 Pay-per-use pricing (no idle costs)
- 🎭 Support for single and multi-person conversations
- 🎯 Two audio modes: Sequential (add) or Simultaneous (para) speaking
- 🎨 Works with cartoon characters and singing
- ⚡ GPU-accelerated inference with Flash Attention 2.6.1
- ☁️ S3 integration for input/output storage
- 🔧 CUDA 12.1 optimized environment
- 🗣️ Label Rotary Position Embedding (L-RoPE) for correct speaker-audio binding
- 🎨 Color correction support to prevent brightness issues

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

### Quick Start

#### Single Person Video
Generate a single-person talking head video:

```bash
modal run app_multitalk_cuda.py
```

This uses default files from your S3 bucket:
- Image: `multi1.png` 
- Audio: `1.wav`

#### Multi-Person Conversation
Generate a multi-person conversation video:

```bash
modal run app_multitalk_cuda.py --two-person
```

This uses:
- Image: `multi1.png` (should contain multiple people)
- Audio: `1.wav` and `2.wav` (one for each person)

### Advanced Usage

#### Single Person with Custom Parameters

```python
from app_multitalk_cuda import app, generate_video_cuda

result = generate_video_cuda.remote(
    prompt="A person speaking enthusiastically about AI",
    image_key="portrait.png",      # Your image in S3
    audio_key="speech.wav",         # Your audio in S3
    sample_steps=20                 # 10-50, higher = better quality
)

print(f"Video uploaded to: {result['s3_output']}")
```

#### Multi-Person Conversations

```python
from app_multitalk_cuda import app, generate_multi_person_video

# Two people talking sequentially (one after another)
result = generate_multi_person_video.remote(
    prompt="Two people having a conversation about technology",
    image_key="two_people.png",
    audio_keys=["person1.wav", "person2.wav"],
    sample_steps=20,
    audio_type="add",        # Sequential speaking (default)
    audio_cfg=4.0,          # Audio guide scale (3-5 recommended)
    color_correction=0.7     # Reduce brightness issues (0-1)
)

# Two people talking simultaneously
result = generate_multi_person_video.remote(
    prompt="Two people singing together",
    image_key="duet.png",
    audio_keys=["singer1.wav", "singer2.wav"],
    sample_steps=20,
    audio_type="para"        # Simultaneous speaking
)
```

### Audio Modes Explained

- **"add" mode (Sequential)**: People speak one after another
  - Person 1 speaks first, then person 2
  - Good for interviews, conversations, dialogues
  
- **"para" mode (Parallel)**: People speak at the same time
  - Both speakers talk simultaneously
  - Good for singing duets, overlapping dialogue

### Parameter Guide

| Parameter | Description | Default | Range/Options |
|-----------|-------------|---------|---------------|
| `prompt` | Text description of the scene | Required | Any descriptive text |
| `image_key` | S3 key for reference image | Required | Must be 896x448 pixels |
| `audio_key(s)` | S3 key(s) for audio file(s) | Required | WAV format, any duration |
| `sample_steps` | Quality/speed tradeoff | 20 | 10-50 (10 min for lip sync) |
| `audio_type` | Speaking mode for multi-person | "add" | "add" or "para" |
| `audio_cfg` | Audio guidance strength | 4.0 | 3-5 (higher = better lip sync) |
| `color_correction` | Brightness correction | 0.7 | 0-1 (lower = less correction) |

## Input Requirements

### Images
- **Resolution**: Must be resized to 896x448 pixels (done automatically)
- **Format**: PNG or JPG
- **Content**: 
  - Single person: Clear portrait photo or character image
  - Multiple people: Image containing all speakers clearly visible
  - Position matters: People are mapped left-to-right to audio files

### Audio
- **Format**: WAV file (MP3 not supported)
- **Sample Rate**: Any (automatically resampled to 16kHz)
- **Duration**: Any length (model defaults to 81 frames / ~3.4 seconds)
- **Quality**: Clear speech without background music for best results

### Multi-Person Setup
- **Audio Mapping**: First audio → leftmost person, second audio → next person, etc.
- **Sequential Mode**: Audios play one after another
- **Simultaneous Mode**: All audios play at the same time
- **Max Speakers**: Tested with 2-3 people

## Output

- **Format**: MP4 video
- **Resolution**: 896x448 pixels
- **Frame Rate**: 24 fps
- **Location**: Uploaded to S3 bucket under `outputs/` prefix

## Common Use Cases

### 1. Interview/Podcast
```python
result = generate_multi_person_video.remote(
    prompt="A podcast host interviewing a tech expert",
    image_key="podcast_studio.png",
    audio_keys=["host_questions.wav", "expert_answers.wav"],
    audio_type="add",  # They speak sequentially
    sample_steps=20
)
```

### 2. Music Duet
```python
result = generate_multi_person_video.remote(
    prompt="Two singers performing a duet",
    image_key="singers.png",
    audio_keys=["singer1.wav", "singer2.wav"],
    audio_type="para",  # They sing together
    sample_steps=30    # Higher quality for music
)
```

### 3. Educational Content
```python
result = generate_video_cuda.remote(
    prompt="A teacher explaining mathematics enthusiastically",
    image_key="teacher.png",
    audio_key="math_lesson.wav",
    sample_steps=20
)
```

## Performance Tips

1. **Optimal Settings**:
   - Sample steps: 20 for normal use, 10 for quick tests, 40+ for high quality
   - Audio CFG: 4.0 works well for most cases, increase to 5.0 for better lip sync
   - Color correction: 0.7 prevents over-brightness, adjust if needed

2. **Processing Time**:
   - Single person (81 frames): ~2-3 minutes on A100
   - Two people (81 frames): ~3-5 minutes on A100
   - Longer videos scale linearly

3. **Cost Optimization**:
   - Use `sample_steps=10` for testing
   - A10G GPUs are more cost-effective than A100
   - Videos are cached for 15 minutes after generation

## Architecture

The implementation uses:
- **Base Image**: NVIDIA CUDA 12.1 development environment
- **PyTorch**: 2.4.1 with CUDA 12.1 support
- **Flash Attention**: 2.6.1 for efficient attention computation
- **xformers**: 0.0.28 for memory-efficient transformers
- **Modal Volumes**: Persistent storage for 82GB of model weights

## Troubleshooting

### Common Issues

1. **Both characters speak with the same audio**
   - Solution: Use `audio_type="add"` for sequential speaking
   - Ensure audio files are different for each person

2. **Characters speak simultaneously when they shouldn't**
   - Solution: Use `audio_type="add"` instead of `"para"`
   - "add" = sequential, "para" = simultaneous

3. **Bright/washed out faces**
   - Solution: Adjust `color_correction` parameter (try 0.5-0.8)
   - Lower values = less correction, darker output

4. **Poor lip sync**
   - Solution: Increase `audio_cfg` to 5.0
   - Ensure audio is clear speech without background music
   - Use at least 10 sample steps

5. **Image dimension errors**
   - Images are automatically resized to 896x448
   - If errors persist, manually resize your image

6. **Memory Issues (OOM)**
   - Reduce sample steps to 10
   - Use A100 GPU instead of A10G
   - Process shorter audio clips

### Best Practices

1. **For Conversations**: Use `audio_type="add"` so people speak in turns
2. **For Singing**: Use `audio_type="para"` for simultaneous vocals
3. **For Quality**: Use 20+ sample steps and `audio_cfg=4.0`
4. **For Testing**: Use 10 sample steps to iterate quickly

## Development

For development setup and contributing guidelines, see [DEVELOPMENT.md](DEVELOPMENT.md).

## Complete Examples

### Example 1: Basic Single Person
```python
from app_multitalk_cuda import app, generate_video_cuda

with app.run():
    result = generate_video_cuda.remote(
        prompt="A professional news anchor delivering breaking news",
        image_key="anchor.png",
        audio_key="news_report.wav",
        sample_steps=20
    )
    print(f"Video ready: {result['s3_output']}")
```

### Example 2: Two-Person Interview (Sequential)
```python
from app_multitalk_cuda import app, generate_multi_person_video

with app.run():
    result = generate_multi_person_video.remote(
        prompt="A journalist interviewing a scientist about climate change",
        image_key="interview_setup.png",
        audio_keys=["journalist_questions.wav", "scientist_answers.wav"],
        audio_type="add",      # They speak one after another
        sample_steps=20,
        audio_cfg=4.0,        # Good lip sync
        color_correction=0.7   # Prevent over-brightness
    )
    print(f"Interview video: {result['s3_output']}")
```

### Example 3: Singing Duet (Simultaneous)
```python
from app_multitalk_cuda import app, generate_multi_person_video

with app.run():
    result = generate_multi_person_video.remote(
        prompt="Two singers performing a beautiful duet on stage",
        image_key="singers_on_stage.png",
        audio_keys=["singer1_part.wav", "singer2_part.wav"],
        audio_type="para",     # They sing at the same time
        sample_steps=30,       # Higher quality for music
        audio_cfg=5.0,        # Maximum lip sync accuracy
        color_correction=0.6   # Adjust for stage lighting
    )
    print(f"Duet video: {result['s3_output']}")
```

### Example 4: Quick Test Mode
```python
# Minimal settings for rapid testing
result = generate_multi_person_video.remote(
    prompt="Two friends chatting",
    image_key="friends.png",
    audio_keys=["friend1.wav", "friend2.wav"],
    sample_steps=10,  # Minimum for decent quality
    audio_type="add"
)
```

## Files

- `app_multitalk_cuda.py` - Main Modal application with Flash Attention and multi-person support
- `s3_utils.py` - S3 integration utilities
- `MODAL_ML_LESSONS_LEARNED.md` - Detailed insights from the implementation journey
- `README.md` - This documentation

## License

This project follows the license terms of the original [MeiGen-MultiTalk](https://github.com/MeiGen-AI/MultiTalk) project.