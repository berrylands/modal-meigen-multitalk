#!/usr/bin/env python3
"""
Minimal test to diagnose the exact issue.
"""

import modal
import os

modal.enable_output()

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

minimal_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["git", "ffmpeg"])
    .pip_install(
        "torch==2.4.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers==4.49.0",
        "diffusers>=0.30.0",
        "librosa",
        "numpy==1.24.4",
        "numba==0.59.1",
        "boto3",
        "Pillow",
        "opencv-python",
        "soundfile",
    )
    .run_commands(
        "cd /root && git clone https://github.com/MeiGen-AI/MultiTalk.git",
    )
)

app = modal.App("test-minimal")
model_volume = modal.Volume.from_name("multitalk-models", create_if_missing=True)

@app.function(
    image=minimal_image,
    gpu="t4",  # Cheaper for testing
    volumes={"/models": model_volume},
    timeout=300,
)
def diagnose_issue():
    """
    Diagnose what's happening.
    """
    import subprocess
    import sys
    import os
    
    print("Diagnostic test...")
    
    # Check if we can run generate_multitalk.py at all
    print("\n1. Testing basic script execution:")
    sys.path.insert(0, "/root/MultiTalk")
    
    # Try with just --help
    result = subprocess.run(
        ["python3", "/root/MultiTalk/generate_multitalk.py", "--help"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ Script runs with --help")
        print("\nHelp output:")
        print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
    else:
        print("❌ Script fails even with --help")
        print(f"Error: {result.stderr[:500]}")
    
    # Check what arguments it expects
    print("\n2. Checking for frame_num in help:")
    if "--frame_num" in result.stdout:
        print("✅ --frame_num IS in the help (expected)")
    else:
        print("⚠️  --frame_num NOT in help")
    
    # Check default values
    print("\n3. Looking for default frame_num:")
    try:
        with open("/root/MultiTalk/generate_multitalk.py", "r") as f:
            content = f.read()
            
        # Look for frame_num default
        import re
        frame_match = re.search(r'--frame_num.*?default[\s=]+(\d+)', content)
        if frame_match:
            default_frames = frame_match.group(1)
            print(f"✅ Default frame_num: {default_frames}")
        else:
            print("⚠️  No default frame_num found")
            
        # Look for size parameter
        if "--size" in content:
            size_match = re.search(r'--size.*?default[\s=]+["\']([^"\']*)["\'\)]', content)
            if size_match:
                print(f"Default size: {size_match.group(1)}")
    except Exception as e:
        print(f"Error reading script: {e}")
    
    # Test with minimal command
    print("\n4. Testing minimal inference command:")
    
    # Create minimal test files
    test_dir = "/tmp/test_minimal"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create test image
    from PIL import Image
    import numpy as np
    
    # Try 480p resolution (854x480)
    img = Image.new('RGB', (854, 480), color='white')
    img.save(f"{test_dir}/test.png")
    
    # Create test audio (16kHz, 3.4 seconds for 81 frames at 24fps)
    import soundfile as sf
    duration = 81 / 24.0  # ~3.375 seconds
    samples = int(16000 * duration)
    audio = np.zeros(samples, dtype=np.float32)
    sf.write(f"{test_dir}/test.wav", audio, 16000)
    
    # Create input JSON
    import json
    input_data = {
        "prompt": "A person is speaking",
        "cond_image": f"{test_dir}/test.png",
        "cond_audio": {"person1": f"{test_dir}/test.wav"}
    }
    
    with open(f"{test_dir}/input.json", "w") as f:
        json.dump(input_data, f)
    
    # Try minimal command
    cmd = [
        "python3", "/root/MultiTalk/generate_multitalk.py",
        "--ckpt_dir", "/models/base",
        "--wav2vec_dir", "/models/wav2vec",
        "--input_json", f"{test_dir}/input.json",
        "--save_file", f"{test_dir}/output",
        "--sample_steps", "5",  # Very few steps
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=test_dir)
    
    if result.returncode == 0:
        print("✅ Minimal command works!")
    else:
        print("❌ Minimal command fails")
        if "frame" in result.stderr:
            print("Frame-related error found")
        if "shape" in result.stderr:
            print("Shape error found")
        print(f"\nError snippet: {result.stderr[-500:]}")
    
    return {"diagnosed": True}


if __name__ == "__main__":
    with app.run():
        print("Running diagnostics...\n")
        result = diagnose_issue.remote()
        print(f"\nDiagnostics complete: {result}")
