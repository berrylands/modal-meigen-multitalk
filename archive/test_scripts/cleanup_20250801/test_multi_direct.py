#!/usr/bin/env python3
"""
Direct test for multi-person generation.
"""

import modal
from app_multitalk_cuda import app, multitalk_cuda_image, model_volume, hf_cache_volume

@app.function(
    image=multitalk_cuda_image,
    gpu="a100-40gb",
    volumes={
        "/models": model_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
    secrets=[
        modal.Secret.from_name("aws-secret"),
        modal.Secret.from_name("huggingface-secret"),
    ],
    timeout=900,
)
def run_two_person_test():
    """
    Direct test function for two-person conversation.
    """
    from app_multitalk_cuda import generate_multi_person_video
    
    print("\n" + "="*60)
    print("🎭 Testing TWO-PERSON Conversation Generation")
    print("="*60)
    print("\nUsing:")
    print("  - Image: multi1.png")
    print("  - Audio 1: 1.wav")
    print("  - Audio 2: 2.wav")
    print("  - Sample steps: 10 (reduced for faster testing)")
    print("\n")
    
    # Call the function directly
    result = generate_multi_person_video.local(
        prompt="Two people having an animated conversation",
        image_key="multi1.png",
        audio_keys=["1.wav", "2.wav"],
        sample_steps=10,  # Reduced for faster testing
        output_prefix="multitalk_2person_test"
    )
    
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    
    if result.get("success"):
        print("🎉 SUCCESS! Two-person video generated!")
        print(f"\n✅ S3 Output: {result['s3_output']}")
        print(f"📊 Size: {result['video_size']:,} bytes")
        print(f"🎬 Frames: {result['frame_count']}")
        print(f"👥 Speakers: {result['num_speakers']}")
        print(f"🎵 Audio durations: {result.get('audio_durations')}")
        print(f"⏱️ Target duration: {result.get('target_duration', 0):.2f}s")
        print(f"🖥️ GPU: {result['gpu']}")
        print(f"⚡ Attention: {result['attention']}")
    else:
        print("❌ FAILED")
        print(f"\nError: {result.get('error', 'Unknown')}")
        if 'stderr' in str(result.get('error', '')):
            print("\nDETAILED ERROR:")
            print("-" * 50)
            print(result.get('error', '')[-2000:])
    
    return result


if __name__ == "__main__":
    with app.run():
        result = run_two_person_test.remote()
        print("\nTest completed.")