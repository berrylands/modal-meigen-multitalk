#!/usr/bin/env python3
"""
Test different audio modes to fix simultaneous speaking.
"""

# ARCHIVED - DO NOT USE
# import modal
# ARCHIVED - DO NOT USE
# from app_multitalk_cuda import app, generate_multi_person_video

# @app.function()
def test_audio_modes():
    """Test 'add' vs 'para' audio modes."""
    
    print("Testing different audio modes for multi-person conversation...\n")
    
    # Test 1: 'add' mode (additive - might handle sequential speaking)
    print("Test 1: 'add' mode with bounding boxes")
    result_add = generate_multi_person_video.local(
        prompt="Two people having a conversation, taking turns to speak",
        image_key="multi1.png",
        audio_keys=["1.wav", "2.wav"],
        sample_steps=10,  # More steps for better quality
        output_prefix="test_add_mode",
        audio_type="add",  # Additive mode
        use_bbox=True
    )
    
    # Test 2: 'para' mode without bbox (original)
    print("\nTest 2: 'para' mode without bounding boxes")
    result_para_no_bbox = generate_multi_person_video.local(
        prompt="Two people having a conversation",
        image_key="multi1.png",
        audio_keys=["1.wav", "2.wav"],
        sample_steps=10,
        output_prefix="test_para_no_bbox",
        audio_type="para",
        use_bbox=False
    )
    
    return {
        "add_with_bbox": result_add.get("success", False),
        "para_no_bbox": result_para_no_bbox.get("success", False),
        "add_s3": result_add.get("s3_output", ""),
        "para_s3": result_para_no_bbox.get("s3_output", "")
    }


if __name__ == "__main__":
    with app.run():
        results = test_audio_modes.remote()
        print("\nResults:")
        print(f"  ADD mode with bbox: {'✅' if results['add_with_bbox'] else '❌'}")
        print(f"  PARA mode no bbox: {'✅' if results['para_no_bbox'] else '❌'}")
        
        if results['add_with_bbox']:
            print(f"\n  ADD output: {results['add_s3']}")
        if results['para_no_bbox']:
            print(f"  PARA output: {results['para_s3']}")