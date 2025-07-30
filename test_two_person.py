#!/usr/bin/env python3
"""
Test two-person conversation generation.
"""

import modal
from app_multitalk_cuda import app, generate_multi_person_video

if __name__ == "__main__":
    with app.run():
        print("="*60)
        print("🎭 Testing TWO-PERSON Conversation Generation")
        print("="*60)
        print("\nUsing:")
        print("  - Image: multi1.png (with multiple people)")
        print("  - Audio 1: 1.wav (315KB)")
        print("  - Audio 2: 2.wav (544KB)")
        print("\n")
        
        result = generate_multi_person_video.remote(
            prompt="Two people having an animated conversation",
            image_key="multi1.png",
            audio_keys=["1.wav", "2.wav"],
            sample_steps=20,
            output_prefix="multitalk_2person_test"
        )
        
        print("\n" + "="*60)
        if result.get("success"):
            print("🎆 SUCCESS! Two-person video generated!")
            print(f"\n✅ S3 Output: {result['s3_output']}")
            print(f"💾 Local: {result.get('local_output')}")
            print(f"📊 Size: {result['video_size']:,} bytes")
            print(f"🎬 Frames: {result['frame_count']}")
            print(f"👥 Speakers: {result['num_speakers']}")
            
            if result.get('audio_durations'):
                print(f"🎵 Audio durations: {[f'{d:.2f}s' for d in result['audio_durations']]}")
            print(f"⏱️ Target duration: {result.get('target_duration', 0):.2f}s")
            
            print(f"🖼️ Image: {result['image_original']} -> {result['image_resized']}")
            print(f"🖥️ GPU: {result['gpu']}")
            print(f"⚡ Attention: {result['attention']}")
            
            print("\n🎉 Multi-person conversation successfully generated!")
            print("\nTo download from S3:")
            print(f"aws s3 cp {result['s3_output']} ./two_person_test.mp4")
        else:
            print("❌ Failed to generate two-person video")
            print(f"\nError: {result.get('error', 'Unknown')[:1000]}")
            if result.get('stdout'):
                print(f"\nSTDOUT: {result['stdout'][:500]}")
        print("="*60)