# Lessons Learned: Running ML Inference Models on Modal

This guide captures key insights from implementing MeiGen-MultiTalk on Modal, which can help accelerate similar ML model deployments.

## 🎯 Key Takeaways

### 1. **Start with the Working Reference Implementation**
- Always examine the original implementation (e.g., Colab notebook) in detail
- Pay attention to:
  - Exact package versions
  - Directory structures
  - File paths (relative vs absolute)
  - Model weight organization
  - Command-line arguments

### 2. **Image Dimension Requirements are Critical**
- Many models have strict input dimension requirements
- In our case: 896x448 pixels was required, not arbitrary resolutions
- The error messages may not clearly indicate dimension issues
- Always check model architecture constraints early

### 3. **Flash Attention Installation Pattern**
```python
# Working pattern for Flash Attention on Modal
multitalk_cuda_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04",
        add_python="3.10"
    )
    .apt_install(["git", "ffmpeg", "libsm6", "libxext6", "wget"])
    .pip_install("torch==2.4.1", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("ninja", "packaging", "wheel", "setuptools")
    .run_commands("pip install flash-attn==2.6.1 --no-build-isolation")
)
```

Key insights:
- Use CUDA development base image (not runtime)
- Install build tools before flash-attn
- Use `--no-build-isolation` flag
- Order matters: PyTorch → build tools → flash-attn

### 4. **Frame Count Constraints**
- Video generation models often have mathematical constraints
- MultiTalk required frame counts following 4n+1 pattern (21, 45, 81, 121...)
- This wasn't documented but was discovered through debugging

### 5. **Working Directory Context Matters**
```python
# Colab pattern (runs from inside repo)
python generate_multitalk.py --ckpt_dir weights/...

# Modal pattern (runs from temp directory)
python /root/MultiTalk/generate_multitalk.py --ckpt_dir /models/...
```

Solutions:
- Either change to the repo directory before execution
- Or use absolute paths everywhere
- Be consistent with your approach

### 6. **Model Weight Setup**
Many models require specific weight file organization:
```python
# Example from MultiTalk
shutil.move("/models/base/diffusion_pytorch_model.safetensors.index.json", 
            "/models/base/old_index.json")
shutil.copy("/models/multitalk/diffusion_pytorch_model.safetensors.index.json", 
            "/models/base/")
shutil.copy("/models/multitalk/multitalk.safetensors", 
            "/models/base/")
```

### 7. **Dependency Resolution Strategy**
1. Start with requirements.txt but expect issues
2. Common problems:
   - NumPy version conflicts (e.g., Numba requires ≤1.26)
   - Missing specialized packages (e.g., misaki for G2P)
   - CUDA compatibility issues
3. Build incrementally and test imports

### 8. **S3 Integration Best Practices**
```python
class S3Manager:
    def __init__(self, bucket_name: str = None):
        self.bucket_name = bucket_name or os.environ.get('AWS_BUCKET_NAME')
        self.s3_client = boto3.client('s3')
```

Tips:
- Use Modal secrets for AWS credentials
- Implement both download and upload in the same session
- Handle large files with streaming
- Clean up temporary files

### 9. **Debugging Strategies**

#### Import Testing
```python
@app.function(image=multitalk_image)
def test_imports():
    try:
        import torch
        import transformers
        # Test all critical imports
        return {"status": "success"}
    except Exception as e:
        return {"error": str(e)}
```

#### Shape Debugging
```python
# Add shape printing at key points
print(f"Input shape: {tensor.shape}")
print(f"Expected: {expected_shape}")
```

#### Environment Verification
```python
def test_environment():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Files in directory: {os.listdir('.')}")
```

### 10. **Modal-Specific Patterns**

#### Volume Mounts for Models
```python
model_volume = modal.Volume.from_name("model-weights", create_if_missing=True)

@app.function(volumes={"/models": model_volume})
def inference():
    # Models persist across runs
```

#### GPU Memory Parameters
```python
GPU_VRAM_PARAMS = {
    "NVIDIA A10G": 8000000000,
    "NVIDIA A100-SXM4-40GB": 11000000000,
}
gpu_name = torch.cuda.get_device_name(0)
vram_param = GPU_VRAM_PARAMS.get(gpu_name, 8000000000)
```

### 11. **Common Pitfalls to Avoid**

1. **Don't assume package availability**
   - Even common packages might not be in the base image
   - Always explicitly install what you need

2. **Don't ignore exact versions**
   - ML models are often sensitive to package versions
   - Match the reference implementation exactly

3. **Don't skip error messages**
   - "Audio length doesn't match frames" → Frame count constraint
   - "Shape mismatch" → Input dimension issue
   - "Flash attention not available" → Installation issue

4. **Don't use runtime CUDA images for compilation**
   - Use development images for packages that need compilation
   - Runtime images lack necessary headers and tools

### 12. **Testing Workflow**

1. **Test image build**
   ```bash
   modal run app.py --action test-environment
   ```

2. **Test model download**
   ```bash
   modal run app.py --action download-models
   ```

3. **Test with minimal input**
   - Use small test files first
   - Verify output format and dimensions

4. **Test with real data**
   - Only after minimal tests pass
   - Monitor GPU memory usage

### 13. **Project Organization**

```
project/
├── app_main.py           # Main Modal app
├── s3_utils.py          # Reusable S3 utilities
├── requirements.txt     # Base requirements
├── test_assets/         # Small test files
├── tests/              # Organized tests
└── archive/            # Development artifacts
```

### 14. **Documentation Best Practices**

1. **Document non-obvious requirements**
   - Input dimensions
   - Frame count constraints
   - Model weight setup steps

2. **Include working examples**
   - Full command lines
   - Expected outputs
   - Common error solutions

3. **Specify exact versions**
   - Python version
   - CUDA version
   - All critical packages

### 15. **Performance Optimization**

1. **Model loading**
   - Use persistent volumes
   - Load models once, reuse across calls
   - Implement model caching

2. **Batch processing**
   - Process multiple inputs when possible
   - Reuse loaded models

3. **GPU selection**
   - T4 for testing (cheapest)
   - A10G for production (good balance)
   - A100 for heavy workloads

## 🚀 Quick Start Template

For new ML model deployments on Modal:

```python
import modal

app = modal.App("my-ml-model")

# Define image with exact versions
ml_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04")
    .pip_install(
        "torch==2.4.1",
        index_url="https://download.pytorch.org/whl/cu121"
    )
    .pip_install([
        # Your specific requirements
    ])
)

# Model storage
model_volume = modal.Volume.from_name("model-weights", create_if_missing=True)

@app.function(
    image=ml_image,
    gpu="a10g",
    volumes={"/models": model_volume}
)
def inference(input_data):
    # Your inference code
    pass

# Always include test function
@app.function(image=ml_image, gpu="t4")
def test_environment():
    import torch
    print(f"CUDA: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    return {"status": "ready"}
```

## 📝 Summary

The journey from a working Colab notebook to a production Modal deployment involves:
1. Understanding exact requirements (dimensions, frame counts, versions)
2. Careful dependency management
3. Proper CUDA environment setup
4. Thorough testing at each step
5. Clear documentation of discoveries

Most importantly: **When something works in Colab but fails on Modal, the difference is usually in the details** - paths, versions, working directories, or hidden constraints.