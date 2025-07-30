#!/usr/bin/env python3
"""
Test to compare single vs multi-person audio handling.
"""

import modal
from app_multitalk_cuda import app, generate_video_cuda, generate_multi_person_video

@app.function()
def compare_audio_handling():
    """Compare how audio is handled in single vs multi-person."""
    
    print("="*60)
    print("COMPARING SINGLE VS MULTI-PERSON AUDIO HANDLING")
    print("="*60)
    
    # Test 1: Single person with first audio
    print("\nTest 1: Single person with audio 1.wav")
    single1 = generate_video_cuda.local(
        prompt="A person speaking",
        image_key="multi1.png",
        audio_key="1.wav",
        sample_steps=2
    )
    print(f"Result: {'✅' if single1.get('success') else '❌'}")
    
    # Test 2: Single person with second audio  
    print("\nTest 2: Single person with audio 2.wav")
    single2 = generate_video_cuda.local(
        prompt="A person speaking",
        image_key="multi1.png", 
        audio_key="2.wav",
        sample_steps=2
    )
    print(f"Result: {'✅' if single2.get('success') else '❌'}")
    
    # Test 3: Multi-person with both audios
    print("\nTest 3: Multi-person with both audios")
    multi = generate_multi_person_video.local(
        prompt="Two people talking",
        image_key="multi1.png",
        audio_keys=["1.wav", "2.wav"],
        sample_steps=2,
        output_prefix="test_compare"
    )
    print(f"Result: {'✅' if multi.get('success') else '❌'}")
    
    return {
        "single1": single1.get('success', False),
        "single2": single2.get('success', False), 
        "multi": multi.get('success', False)
    }


if __name__ == "__main__":
    with app.run():
        result = compare_audio_handling.remote()
        print(f"\nComparison results:")
        print(f"  Single with audio 1: {result['single1']}")
        print(f"  Single with audio 2: {result['single2']}")
        print(f"  Multi with both: {result['multi']}")