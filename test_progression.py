#!/usr/bin/env python3
"""
Test progression from single to multi-person.
"""

import modal
from app_multitalk_cuda import app, generate_video_cuda, generate_multi_person_video

@app.function()
def test_single_then_multi():
    """Test single person first, then multi-person."""
    
    print("="*60)
    print("STEP 1: Testing SINGLE person generation (5 steps)")
    print("="*60)
    
    # Test single person with minimal steps
    single_result = generate_video_cuda.local(
        prompt="A person speaking",
        image_key="multi1.png",
        audio_key="1.wav",
        sample_steps=5  # Minimal for testing
    )
    
    if single_result.get("success"):
        print("✅ Single person: SUCCESS")
        print(f"   Output: {single_result['s3_output']}")
    else:
        print("❌ Single person: FAILED")
        print(f"   Error: {single_result.get('error', 'Unknown')[:200]}")
        return {"single": "failed", "multi": "skipped"}
    
    print("\n" + "="*60)
    print("STEP 2: Testing MULTI person generation (5 steps)")
    print("="*60)
    
    # Test multi-person with minimal steps
    multi_result = generate_multi_person_video.local(
        prompt="Two people talking",
        image_key="multi1.png",
        audio_keys=["1.wav", "2.wav"],
        sample_steps=5,  # Minimal for testing
        output_prefix="test_multi_minimal"
    )
    
    if multi_result.get("success"):
        print("✅ Multi person: SUCCESS")
        print(f"   Output: {multi_result['s3_output']}")
        print(f"   Speakers: {multi_result['num_speakers']}")
    else:
        print("❌ Multi person: FAILED")
        print(f"   Error: {multi_result.get('error', 'Unknown')[:500]}")
    
    return {
        "single": "success" if single_result.get("success") else "failed",
        "multi": "success" if multi_result.get("success") else "failed"
    }


if __name__ == "__main__":
    with app.run():
        print("\nStarting progression test...\n")
        result = test_single_then_multi.remote()
        
        print("\n" + "="*60)
        print("FINAL RESULTS:")
        print(f"  Single person: {result['single']}")
        print(f"  Multi person: {result['multi']}")
        print("="*60)