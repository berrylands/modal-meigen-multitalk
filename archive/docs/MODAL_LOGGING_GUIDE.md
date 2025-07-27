# Modal Logging and Debugging Guide

## Quick Start - Enable All Debugging

```python
import modal
import os
import sys

# Maximum debugging setup
sys.stdout.reconfigure(line_buffering=True)
modal.enable_output()
os.environ["MODAL_LOGLEVEL"] = "DEBUG"
os.environ["MODAL_TRACEBACK"] = "1"
```

## Configuration Methods

### 1. Environment Variables
```bash
export MODAL_LOGLEVEL=DEBUG
export MODAL_TRACEBACK=1
```

### 2. Modal CLI Configuration
```bash
modal config set loglevel DEBUG
modal config set traceback true
modal config show  # Verify settings
```

### 3. In Python Code
```python
modal.enable_output()  # MUST be called to see build logs
```

## CLI Commands for Logs

### View App Logs
```bash
# List all apps
modal app list

# Get logs for specific app
modal app logs <app-id-or-name>

# With timestamps
modal app logs <app-id> --timestamps
```

### Container Logs
```bash
# List containers
modal container list

# View container logs
modal container logs <container-id>
```

## Build Log Visibility

### What We Learned

1. **Build logs ARE visible** when `modal.enable_output()` is used
2. The actual error in our case was the environment variable syntax:
   ```python
   # ❌ Wrong - Modal doesn't support shell expansion
   .env({"PYTHONPATH": "/root/MultiTalk:$PYTHONPATH"})
   
   # ✅ Correct
   .env({"PYTHONPATH": "/root/MultiTalk"})
   ```

3. Build logs show:
   - Each build step
   - Package downloads and installations
   - Error messages with details
   - Build timings

### Example Build Output
```
Building image im-L2tvOiFSZe3O2bcjAsDmZB

=> Step 0: FROM base
=> Step 1: RUN python -m pip install transformers==4.49.0
Looking in indexes: http://pypi-mirror.modal.local:5555/simple
Collecting transformers==4.49.0
...
Successfully installed transformers-4.49.0
Saving image...
Image saved, took 3.37s
Built image im-L2tvOiFSZe3O2bcjAsDmZB in 19.44s
```

## Debug Pattern for Image Builds

```python
"""Debug pattern for Modal image builds"""
import modal
import os
import sys

# Enable all debugging
sys.stdout.reconfigure(line_buffering=True)
modal.enable_output()
os.environ["MODAL_LOGLEVEL"] = "DEBUG"
os.environ["MODAL_TRACEBACK"] = "1"

# Your image definition
test_image = modal.Image.debian_slim(python_version="3.10")

app = modal.App("debug-test")

@app.function(image=test_image)
def test():
    return "Success"

if __name__ == "__main__":
    try:
        with modal.enable_output():  # Double-ensure output
            with app.run():
                result = test.remote()
                print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}")
        if hasattr(e, '__cause__'):
            print(f"Cause: {e.__cause__}")
        raise  # Re-raise for full traceback
```

## Common Issues and Solutions

### 1. No Build Logs Visible
**Solution**: Always call `modal.enable_output()` before building

### 2. Environment Variable Errors
**Issue**: `failed to resolve variables for string: /path:$VAR`
**Solution**: Modal doesn't support shell variable expansion in `.env()`

### 3. Silent Failures
**Solution**: Enable debug logging and traceback:
```bash
modal config set loglevel DEBUG
modal config set traceback true
```

### 4. Package Installation Failures
Build logs will show the exact pip error, making debugging straightforward

## Key Findings

1. **Modal DOES provide build logs** - just need to enable output
2. **Debug logging is comprehensive** - shows all build steps
3. **Environment variable syntax is strict** - no shell expansion
4. **Modal uses a local PyPI mirror** for faster builds
5. **Build caching works well** - subsequent builds are much faster

## Best Practices

1. Always use `modal.enable_output()` during development
2. Set debug logging when troubleshooting
3. Check syntax carefully for environment variables
4. Use the Modal dashboard as a backup for viewing logs
5. Keep build steps simple and incremental for easier debugging