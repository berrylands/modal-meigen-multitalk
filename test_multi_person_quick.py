#!/usr/bin/env python3
"""
Quick test for multi-person generation with reduced frames.
"""

import modal
from app_multitalk_cuda import app, generate_multi_person_video

if __name__ == "__main__":
    with app.run():
        print("="*60)
        print("🎭 Quick Multi-Person Test (10 sample steps)")
        print("="*60)
        
        result = generate_multi_person_video.remote(
            prompt="Two people having a quick conversation",
            image_key="multi1.png",
            audio_keys=["1.wav", "2.wav"],
            sample_steps=10,  # Reduced for faster testing
            output_prefix="multitalk_quick_test"
        )
        
        print("\n" + "="*60)
        if result.get("success"):
            print("✅ SUCCESS!")
            print(f"S3 Output: {result['s3_output']}")
            print(f"Speakers: {result['num_speakers']}")
            print(f"Audio durations: {result.get('audio_durations')}")
        else:
            print("❌ FAILED")
            print(f"Error: {result.get('error', 'Unknown')[:1000]}")
        print("="*60)