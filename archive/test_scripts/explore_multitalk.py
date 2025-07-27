"""
Explore MultiTalk repository structure and dependencies in Modal container.
"""

import modal

app = modal.App("explore-multitalk")

# Define the image inline
multitalk_image_light = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install([
        "git",
        "ffmpeg",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
        "wget",
        "build-essential",
        "ninja-build",
    ])
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1", 
        "torchaudio==2.4.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "xformers==0.0.28",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers==4.49.0",
        "peft",
        "accelerate",
        "huggingface_hub",
        "ninja",
        "psutil",
        "packaging",
        "librosa",
        "moviepy",
        "opencv-python",
        "Pillow",
        "diffusers>=0.30.0",
        "numpy==1.26.4",
        "numba==0.59.1",
        "boto3",
        "tqdm",
        "scipy",
        "soundfile",
    )
    .run_commands(
        "cd /root && git clone https://github.com/MeiGen-AI/MultiTalk.git",
        "cd /root/MultiTalk && pip install -r requirements.txt || true",
    )
    .env({
        "PYTHONPATH": "/root/MultiTalk",
        "TORCH_CUDA_ARCH_LIST": "7.0;7.5;8.0;8.6;8.9;9.0",
        "CUDA_VISIBLE_DEVICES": "0",
    })
)

@app.function(
    image=multitalk_image_light,
    gpu="t4",  # Use cheaper GPU for exploration
)
def explore_multitalk_repo():
    """Explore the MultiTalk repository structure and dependencies."""
    import os
    import subprocess
    import sys
    
    print("="*60)
    print("MultiTalk Repository Exploration")
    print("="*60)
    
    # Check if MultiTalk repo exists
    multitalk_path = "/root/MultiTalk"
    if not os.path.exists(multitalk_path):
        print(f"ERROR: MultiTalk repo not found at {multitalk_path}")
        return {"error": "MultiTalk repo not found"}
    
    print(f"\n✅ MultiTalk repo found at: {multitalk_path}")
    
    # List repository structure
    print("\n📁 Repository Structure:")
    for root, dirs, files in os.walk(multitalk_path):
        level = root.replace(multitalk_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        
        # Limit depth to avoid too much output
        if level < 3:
            subindent = ' ' * 2 * (level + 1)
            for file in files[:10]:  # Limit files shown
                print(f'{subindent}{file}')
            if len(files) > 10:
                print(f'{subindent}... and {len(files) - 10} more files')
    
    # Check for requirements files
    print("\n📋 Requirements Files:")
    req_files = ["requirements.txt", "requirements.in", "setup.py", "pyproject.toml"]
    for req_file in req_files:
        req_path = os.path.join(multitalk_path, req_file)
        if os.path.exists(req_path):
            print(f"\n  Found: {req_file}")
            with open(req_path, 'r') as f:
                content = f.read()
                print("  Content:")
                lines = content.split('\n')
                print("  " + "\n  ".join(lines[:20]))  # First 20 lines
                if len(lines) > 20:
                    print(f"  ... and {len(lines) - 20} more lines")
    
    # Search for 'misaki' mentions
    print("\n🔍 Searching for 'misaki' in the codebase:")
    try:
        result = subprocess.run(
            ["grep", "-r", "-i", "misaki", multitalk_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.stdout:
            print("  Found mentions:")
            lines = result.stdout.split('\n')[:10]  # First 10 matches
            for line in lines:
                print(f"  {line}")
            total_lines = len(result.stdout.split('\n'))
            if total_lines > 10:
                print(f"  ... and {total_lines - 10} more matches")
        else:
            print("  No mentions of 'misaki' found")
    except subprocess.TimeoutExpired:
        print("  Search timed out")
    
    # Check kokoro/pipeline.py specifically
    print("\n📄 Checking kokoro/pipeline.py:")
    kokoro_path = os.path.join(multitalk_path, "kokoro/pipeline.py")
    if os.path.exists(kokoro_path):
        print(f"  Found at: {kokoro_path}")
        with open(kokoro_path, 'r') as f:
            content = f.read()
            # Look for imports
            import_lines = [line for line in content.split('\n') if 'import' in line]
            print("  Import statements:")
            for line in import_lines[:20]:
                print(f"    {line}")
    else:
        # Try to find kokoro directory
        print("  kokoro/pipeline.py not found, searching for kokoro directory:")
        for root, dirs, files in os.walk(multitalk_path):
            if 'kokoro' in dirs or 'kokoro' in root:
                print(f"    Found: {root}")
                # List files in kokoro
                kokoro_dir = os.path.join(root, 'kokoro') if 'kokoro' in dirs else root
                if os.path.exists(kokoro_dir):
                    files = os.listdir(kokoro_dir)
                    print(f"    Files: {files[:10]}")
    
    # Check if there are any custom modules in the repo
    print("\n📦 Custom Python modules in MultiTalk:")
    for root, dirs, files in os.walk(multitalk_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, multitalk_path)
                # Check if it might be a module named misaki
                if 'misaki' in file.lower() or 'misaki' in root.lower():
                    print(f"  Potential match: {rel_path}")
    
    # List all Python files that might be modules
    print("\n🐍 Python modules (first 20):")
    py_files = []
    for root, dirs, files in os.walk(multitalk_path):
        for file in files:
            if file.endswith('.py') and not file.startswith('test_'):
                rel_path = os.path.relpath(os.path.join(root, file), multitalk_path)
                py_files.append(rel_path)
    
    for py_file in sorted(py_files)[:20]:
        print(f"  {py_file}")
    if len(py_files) > 20:
        print(f"  ... and {len(py_files) - 20} more Python files")
    
    return {"status": "complete", "files_found": len(py_files)}

@app.function(
    image=multitalk_image_light,
    gpu="t4",
)
def check_missing_dependencies():
    """Try to import modules and see what's missing."""
    import sys
    import subprocess
    
    print("="*60)
    print("Checking Missing Dependencies")
    print("="*60)
    
    # Add MultiTalk to path
    sys.path.insert(0, "/root/MultiTalk")
    
    # Try to import key modules
    modules_to_check = [
        "kokoro",
        "kokoro.pipeline",
        "misaki",
        "generate_multitalk",
    ]
    
    for module_name in modules_to_check:
        print(f"\n🔍 Trying to import: {module_name}")
        try:
            if '.' in module_name:
                # Handle submodules
                parts = module_name.split('.')
                module = __import__(parts[0])
                for part in parts[1:]:
                    module = getattr(module, part)
            else:
                module = __import__(module_name)
            print(f"  ✅ Success! Module location: {getattr(module, '__file__', 'built-in')}")
        except ImportError as e:
            print(f"  ❌ ImportError: {e}")
            # Try to get more info
            if "misaki" in str(e):
                print("  🔍 Looking for misaki-related files...")
                result = subprocess.run(
                    ["find", "/root/MultiTalk", "-name", "*misaki*", "-o", "-name", "*Misaki*"],
                    capture_output=True,
                    text=True
                )
                if result.stdout:
                    print("  Found files:")
                    print("  " + result.stdout)
        except Exception as e:
            print(f"  ❌ Error: {type(e).__name__}: {e}")
    
    # Try running generate_multitalk.py with help to see dependencies
    print("\n📋 Checking generate_multitalk.py requirements:")
    try:
        result = subprocess.run(
            ["python", "/root/MultiTalk/generate_multitalk.py", "--help"],
            capture_output=True,
            text=True,
            cwd="/root/MultiTalk"
        )
        if result.returncode != 0:
            print(f"  Error running generate_multitalk.py:")
            print(f"  STDOUT: {result.stdout}")
            print(f"  STDERR: {result.stderr}")
    except Exception as e:
        print(f"  Exception: {e}")
    
    return {"status": "complete"}

@app.local_entrypoint()
def main():
    """Run exploration functions."""
    print("Exploring MultiTalk repository...")
    
    # First explore the repository structure
    result1 = explore_multitalk_repo.remote()
    print(f"\nExploration result: {result1}")
    
    # Then check missing dependencies
    print("\n" + "="*60)
    print("Now checking missing dependencies...")
    result2 = check_missing_dependencies.remote()
    print(f"\nDependency check result: {result2}")

if __name__ == "__main__":
    main()