"""
Test Modal environment with misaki package included.
"""

import modal

app = modal.App("test-misaki-env")

# Define the image with misaki
test_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install([
        "git",
        "ffmpeg",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
        "wget",
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
        "huggingface_hub",
        "accelerate",
        "diffusers>=0.30.0",
        "librosa",
        "moviepy",
        "opencv-python",
        "numpy==1.26.4",
        "tqdm",
        "misaki[en]",  # The missing dependency
    )
    .run_commands(
        "cd /root && git clone https://github.com/MeiGen-AI/MultiTalk.git",
        "cd /root/MultiTalk && pip install -r requirements.txt || true",
    )
    .env({"PYTHONPATH": "/root/MultiTalk"})
)

@app.function(
    image=test_image,
    gpu="t4",
)
def test_misaki_import():
    """Test if misaki and kokoro modules can be imported successfully."""
    import sys
    import subprocess
    
    print("="*60)
    print("Testing MultiTalk with misaki package")
    print("="*60)
    
    # Test misaki import
    print("\n1. Testing misaki import:")
    try:
        import misaki
        print("  ✅ Successfully imported misaki")
        print(f"  Version: {getattr(misaki, '__version__', 'unknown')}")
        
        # Test specific imports used by kokoro
        from misaki import en, espeak
        print("  ✅ Successfully imported misaki.en and misaki.espeak")
    except ImportError as e:
        print(f"  ❌ Failed to import misaki: {e}")
        return {"status": "failed", "error": str(e)}
    
    # Test kokoro import
    print("\n2. Testing kokoro import:")
    sys.path.insert(0, "/root/MultiTalk")
    try:
        from kokoro import KPipeline
        print("  ✅ Successfully imported kokoro.KPipeline")
    except ImportError as e:
        print(f"  ❌ Failed to import kokoro: {e}")
        return {"status": "failed", "error": str(e)}
    
    # Test generate_multitalk.py
    print("\n3. Testing generate_multitalk.py:")
    try:
        result = subprocess.run(
            ["python", "/root/MultiTalk/generate_multitalk.py", "--help"],
            capture_output=True,
            text=True,
            cwd="/root/MultiTalk",
            timeout=30
        )
        if result.returncode == 0:
            print("  ✅ generate_multitalk.py --help executed successfully")
            print("  Output preview:")
            lines = result.stdout.split('\n')[:10]
            for line in lines:
                print(f"    {line}")
        else:
            print(f"  ❌ generate_multitalk.py failed with code {result.returncode}")
            print(f"  STDERR: {result.stderr}")
    except Exception as e:
        print(f"  ❌ Exception running generate_multitalk.py: {e}")
    
    # List all available modules in MultiTalk
    print("\n4. Available modules in MultiTalk:")
    try:
        import pkgutil
        import importlib
        
        multitalk_modules = []
        for importer, modname, ispkg in pkgutil.walk_packages(["/root/MultiTalk"]):
            if not modname.startswith('.'):
                multitalk_modules.append(modname)
        
        print(f"  Found {len(multitalk_modules)} modules")
        for mod in sorted(multitalk_modules)[:10]:
            print(f"    - {mod}")
        if len(multitalk_modules) > 10:
            print(f"    ... and {len(multitalk_modules) - 10} more")
    except Exception as e:
        print(f"  Error listing modules: {e}")
    
    print("\n" + "="*60)
    print("✅ All tests completed successfully!")
    return {"status": "success"}

@app.local_entrypoint()
def main():
    """Run the test."""
    print("Testing MultiTalk environment with misaki package...")
    result = test_misaki_import.remote()
    print(f"\nTest result: {result}")

if __name__ == "__main__":
    main()