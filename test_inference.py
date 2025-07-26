"""
Test inference with MeiGen-MultiTalk on Modal.
"""

import modal
import requests
import os
from pathlib import Path

# Download test assets
def download_test_assets():
    """Download test image and audio for inference testing."""
    
    # Create test directory
    test_dir = Path("test_assets")
    test_dir.mkdir(exist_ok=True)
    
    # Test image - a simple portrait
    # Using a public domain image
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/Parque_Estado_MA_2017_%2833%29.jpg/256px-Parque_Estado_MA_2017_%2833%29.jpg"
    image_path = test_dir / "test_portrait.jpg"
    
    if not image_path.exists():
        print("Downloading test image...")
        response = requests.get(image_url)
        with open(image_path, "wb") as f:
            f.write(response.content)
        print(f"  Saved to: {image_path}")
    
    # For audio, we need to create a simple test audio
    # Using Modal to generate it
    audio_path = test_dir / "test_audio.wav"
    
    if not audio_path.exists():
        print("Creating test audio...")
        # Create a simple sine wave as test audio
        import numpy as np
        import scipy.io.wavfile as wavfile
        
        duration = 3.0  # seconds
        sample_rate = 16000
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # Add some modulation to simulate speech-like variation
        modulation = 0.2 * np.sin(2 * np.pi * 3 * t)
        audio = audio * (1 + modulation)
        
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        
        wavfile.write(str(audio_path), sample_rate, audio_int16)
        print(f"  Created audio: {audio_path}")
        print(f"  Duration: {duration}s, Sample rate: {sample_rate}Hz")
    
    return str(image_path), str(audio_path)

def run_inference_test():
    """Run inference test with Modal."""
    
    # Get test assets
    image_path, audio_path = download_test_assets()
    
    print("\nRunning inference test...")
    print(f"Image: {image_path}")
    print(f"Audio: {audio_path}")
    
    # Run using Modal CLI
    import subprocess
    
    try:
        # First, ensure the assets are available
        abs_image_path = os.path.abspath(image_path)
        abs_audio_path = os.path.abspath(audio_path)
        
        print("\nCalling Modal app for inference...")
        cmd = [
            "modal", "run", "modal_meigen_multitalk.py",
            "--action", "generate",
            "--audio-path", abs_audio_path,
            "--image-path", abs_image_path,
            "--prompt", "A person is speaking"
        ]
        
        print(f"Command: {' '.join(cmd)}")
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("\n✅ Modal execution completed successfully!")
            print("Output:", result.stdout)
            
            # Check if output video was created
            if os.path.exists("output_video.mp4"):
                size = os.path.getsize("output_video.mp4") / 1024 / 1024
                print(f"\n✅ Success! Video saved to: output_video.mp4")
                print(f"   Size: {size:.2f} MB")
                return True
            else:
                print("\n❌ No output video found")
                return False
        else:
            print(f"\n❌ Modal execution failed with code {result.returncode}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"\n❌ Inference failed: {type(e).__name__}")
        print(f"   Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("MeiGen-MultiTalk Inference Test")
    print("="*60)
    
    success = run_inference_test()
    
    if success:
        print("\n🎉 Inference test PASSED!")
        print("The complete pipeline is working.")
    else:
        print("\n❌ Inference test FAILED!")
        print("Check the error messages above.")