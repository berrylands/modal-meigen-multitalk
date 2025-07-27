# Detailed Line-by-Line Comparison: Colab vs Modal Implementation

## Key Differences Found

### 1. Working Directory

**Colab:**
- Commands are run from the MultiTalk directory itself
- Command: `python3 generate_multitalk.py` (relative path)
- Working directory: `/content/MultiTalk` or similar

**Modal:**
- Commands are run from a temporary work directory (e.g., `/tmp/multitalk_*`)
- Command: `python3 /root/MultiTalk/generate_multitalk.py` (absolute path)
- Working directory: temporary directory with `cwd=work_dir`

**Impact:** This could affect relative path resolution for model weights and file loading.

### 2. Command Structure

**Colab (from COLAB_INSIGHTS.md):**
```bash
python3 generate_multitalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir weights/chinese-wav2vec2-base \
    --input_json input.json \
    --sample_steps 20 \
    --num_persistent_param_in_dit "{VRAM_PARAM}" \
    --mode streaming \
    --use_teacache \
    --save_file output
```

**Modal (from app_multitalk_exact_colab.py):**
```python
cmd = [
    "python3", "/root/MultiTalk/generate_multitalk.py",
    "--ckpt_dir", "/models/base",
    "--wav2vec_dir", "/models/wav2vec",
    "--input_json", input_json_path,
    "--sample_steps", str(sample_steps),
    "--num_persistent_param_in_dit", str(vram_param),
    "--mode", "streaming",
    "--use_teacache",
    "--save_file", output_path,
]
```

**Key Differences:**
- Colab uses relative paths for weights (`weights/...`)
- Modal uses absolute paths (`/models/...`)
- Input/output paths are different

### 3. Model Weight Paths

**Colab:**
- `weights/Wan2.1-I2V-14B-480P`
- `weights/chinese-wav2vec2-base`
- Weights are in a `weights/` subdirectory relative to MultiTalk

**Modal:**
- `/models/base`
- `/models/wav2vec`
- Weights are in a separate `/models` volume mount

### 4. Environment Setup

**Colab:**
- Runs in the MultiTalk directory
- Has access to relative paths
- PYTHONPATH not explicitly set (runs from within the repo)

**Modal:**
- Sets PYTHONPATH to `/root/MultiTalk`
- Runs from temporary directory
- Uses sys.path.insert(0, "/root/MultiTalk")

### 5. Missing Dependency: misaki

**Issue:** The `misaki` package is imported by `kokoro/pipeline.py` but the import happens at the module level, causing immediate failure.

**Colab:** Likely has `misaki[en]` pre-installed or installs it separately.

**Modal:** Includes `misaki[en]` in pip_install but the import still fails, suggesting:
- The package might not be installing correctly
- There might be a version conflict
- The import might need to happen after proper initialization

### 6. File Organization

**Colab Structure:**
```
/content/MultiTalk/
├── generate_multitalk.py
├── weights/
│   ├── Wan2.1-I2V-14B-480P/
│   ├── chinese-wav2vec2-base/
│   └── MeiGen-MultiTalk/
├── input.json
└── output.mp4
```

**Modal Structure:**
```
/root/MultiTalk/           # Git repo
/models/                   # Volume mount
│   ├── base/
│   ├── wav2vec/
│   └── multitalk/
/tmp/multitalk_*/          # Working directory
│   ├── input.json
│   ├── input.png
│   ├── input.wav
│   └── output.mp4
```

### 7. Input File Handling

**Colab:**
- Image and audio files are likely in the same directory
- Paths in input.json are simple filenames or relative paths

**Modal:**
- Files are downloaded to temp directory
- Absolute paths are used in input.json

### 8. Flash Attention Version

**Colab:** Uses `flash-attn==2.6.1`
**Modal:** Various versions tried, including 2.6.1, but not in main modal_meigen_multitalk.py

### 9. Execution Context

**Colab:**
- Interactive notebook environment
- User can debug step by step
- Working directory persists between cells

**Modal:**
- Batch execution in container
- No interactive debugging
- Fresh temporary directory each run

## Recommendations for Fixing Modal Implementation

1. **Change Working Directory:**
   ```python
   # Instead of running from work_dir
   os.chdir("/root/MultiTalk")
   # Then use relative paths like Colab
   ```

2. **Use Relative Paths for Weights:**
   ```python
   # Create symlinks or copy weights to match Colab structure
   os.makedirs("/root/MultiTalk/weights", exist_ok=True)
   os.symlink("/models/base", "/root/MultiTalk/weights/Wan2.1-I2V-14B-480P")
   os.symlink("/models/wav2vec", "/root/MultiTalk/weights/chinese-wav2vec2-base")
   ```

3. **Fix misaki Import:**
   - Install misaki before other packages that depend on it
   - Or modify the import to be conditional/lazy

4. **Match Exact Environment:**
   - Use exact same pip package versions
   - Ensure CUDA environment matches

5. **Simplify File Paths:**
   - Put input files in MultiTalk directory
   - Use simple filenames in input.json

The core issue appears to be that Modal runs the command from a different directory with different path structures than Colab expects.