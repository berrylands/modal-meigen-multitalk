#!/usr/bin/env python3
"""
Analyze the audio file from S3 to understand MultiTalk requirements.
"""

import modal
import os

modal.enable_output()

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

# Simple image with audio processing tools
audio_test_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "boto3",
        "librosa",
        "numpy==1.24.4",
        "numba==0.59.1",
        "soundfile",
        "scipy",
    )
)

app = modal.App("audio-analysis")

@app.function(
    image=audio_test_image,
    secrets=[modal.Secret.from_name("aws-secret")]
)
def analyze_audio():
    """
    Analyze the audio file from S3 to understand what MultiTalk expects.
    """
    import boto3
    import librosa
    import numpy as np
    import soundfile as sf
    import tempfile
    
    print("="*60)
    print("Audio File Analysis for MultiTalk")
    print("="*60)
    
    # Get bucket
    bucket_name = os.environ.get('AWS_BUCKET_NAME')
    if not bucket_name:
        return {"error": "AWS_BUCKET_NAME not found"}
    
    print(f"\nBucket: {bucket_name}")
    
    try:
        # Download audio from S3
        s3 = boto3.client('s3')
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_path = tmp.name
            s3.download_file(bucket_name, "1.wav", audio_path)
            print(f"✅ Downloaded 1.wav to {audio_path}")
        
        # Analyze with librosa
        print("\n🎵 Audio Analysis:")
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)  # Keep original sample rate
        print(f"  Sample rate: {sr} Hz")
        print(f"  Duration: {len(y)/sr:.2f} seconds")
        print(f"  Total samples: {len(y):,}")
        print(f"  Shape: {y.shape}")
        print(f"  Data type: {y.dtype}")
        print(f"  Min value: {y.min():.4f}")
        print(f"  Max value: {y.max():.4f}")
        
        # Check with soundfile for more info
        print("\n📄 File Info (via soundfile):")
        info = sf.info(audio_path)
        print(f"  Channels: {info.channels}")
        print(f"  Frames: {info.frames:,}")
        print(f"  Samplerate: {info.samplerate} Hz")
        print(f"  Subtype: {info.subtype}")
        print(f"  Format: {info.format}")
        print(f"  Duration: {info.duration:.2f} seconds")
        
        # MultiTalk specific checks
        print("\n🎥 MultiTalk Requirements:")
        print(f"  Default frame_num: 81")
        print(f"  Expected FPS: ~24-30")
        print(f"  Expected video duration: ~2.7-3.4 seconds (81 frames)")
        
        # Calculate if audio matches
        expected_duration = 81 / 24.0  # Assuming 24 FPS
        print(f"\n  Audio duration: {info.duration:.2f}s")
        print(f"  Expected duration: {expected_duration:.2f}s")
        print(f"  Match: {'✅ Close enough' if abs(info.duration - expected_duration) < 1.0 else '❌ Too different'}")
        
        # Resample to 16kHz if needed (common for speech models)
        print("\n🎤 Resampling to 16kHz (common for speech):")
        y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)
        print(f"  New shape: {y_16k.shape}")
        print(f"  New duration: {len(y_16k)/16000:.2f} seconds")
        
        # Save resampled version
        resampled_path = "/tmp/audio_16khz.wav"
        sf.write(resampled_path, y_16k, 16000)
        print(f"  Saved to: {resampled_path}")
        
        # Get file size
        import os as os_module
        orig_size = os_module.path.getsize(audio_path)
        resampled_size = os_module.path.getsize(resampled_path)
        print(f"\n📊 File sizes:")
        print(f"  Original: {orig_size:,} bytes")
        print(f"  Resampled: {resampled_size:,} bytes")
        
        return {
            "success": True,
            "original": {
                "duration": info.duration,
                "sample_rate": sr,
                "channels": info.channels,
                "format": info.format
            },
            "matches_multitalk": abs(info.duration - expected_duration) < 1.0
        }
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    with app.run():
        print("Analyzing audio file from S3...\n")
        
        result = analyze_audio.remote()
        
        print("\n" + "="*60)
        if result.get("success"):
            print("✅ Analysis complete!")
            if result.get("matches_multitalk"):
                print("✅ Audio should work with MultiTalk")
            else:
                print("⚠️  Audio duration might not match MultiTalk expectations")
                print("   MultiTalk might need audio that matches video length")
        else:
            print("❌ Analysis failed!")
            print(f"Error: {result.get('error')}")
        print("="*60)
