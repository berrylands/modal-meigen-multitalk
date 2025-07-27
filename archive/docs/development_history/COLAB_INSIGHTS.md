# Key Insights from MeiGen-MultiTalk Colab Notebook

## Critical Requirements

### 1. GPU Requirements
- **Supported GPUs**: NVIDIA A100 (recommended) or L4 (slower)
- **VRAM Requirements**: 16GB+ VRAM minimum
- **RAM Requirements**: 53GB system RAM
- **Performance**: ~5 minutes per 1 second of video on A100

### 2. Dependency Versions (CRITICAL)
```bash
# CUDA 12.1 versions
torch==2.4.1
torchvision==0.19.1
torchaudio==2.4.1
xformers==0.0.28
flash-attn==2.6.1  # Note: Different from our requirements.txt!
transformers==4.49.0  # KEY: Must be pre-CVE-2025-32434 patch version
```

### 3. Model Files Required
1. **Wan2.1-I2V-14B-480P** - Base model
2. **chinese-wav2vec2-base** - Audio encoder
3. **MeiGen-MultiTalk** - MultiTalk weights

### 4. Model Setup Process
```bash
# Download models
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./weights/Wan2.1-I2V-14B-480P
huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir ./weights/chinese-wav2vec2-base
huggingface-cli download MeiGen-AI/MeiGen-MultiTalk --local-dir ./weights/MeiGen-MultiTalk

# CRITICAL: Copy MultiTalk weights into base model directory
mv weights/Wan2.1-I2V-14B-480P/diffusion_pytorch_model.safetensors.index.json weights/Wan2.1-I2V-14B-480P/diffusion_pytorch_model.safetensors.index.json_old
cp weights/MeiGen-MultiTalk/diffusion_pytorch_model.safetensors.index.json weights/Wan2.1-I2V-14B-480P/
cp weights/MeiGen-MultiTalk/multitalk.safetensors weights/Wan2.1-I2V-14B-480P/
```

### 5. VRAM Optimization Parameters
```python
GPU_TO_VRAM_PARAMS = {
    "NVIDIA A100": 11000000000,
    "NVIDIA A100-SXM4-40GB": 11000000000,
    "NVIDIA A100-SXM4-80GB": 22000000000,
    "NVIDIA L4": 5000000000
}
```

### 6. Inference Command
```bash
python3 generate_multitalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir weights/chinese-wav2vec2-base \
    --input_json input.json \
    --sample_steps 20 \  # 40 for higher quality
    --num_persistent_param_in_dit "{VRAM_PARAM}" \
    --mode streaming \
    --use_teacache \
    --save_file output
```

### 7. Input Format
```json
{
    "prompt": "A man is speaking in a studio.",
    "cond_image": "image.png",
    "cond_audio": {
        "person1": "audio.wav"
    }
}
```

## Key Differences for Modal

1. **flash-attn version**: Colab uses 2.6.1, our requirements has 2.7.4.post1
2. **transformers version**: MUST use 4.49.0 (pre-CVE patch)
3. **System packages**: Need ffmpeg
4. **Clone official repo**: Need MultiTalk source code
5. **CUDA index**: Uses cu121 (CUDA 12.1)

## Action Items for Modal Implementation

1. Update flash-attn to 2.6.1
2. Pin transformers to 4.49.0
3. Clone MultiTalk repo in image build
4. Implement model download and setup
5. Configure VRAM parameters based on GPU type
6. Create inference wrapper