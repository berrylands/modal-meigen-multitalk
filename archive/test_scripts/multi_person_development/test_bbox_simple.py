#!/usr/bin/env python3
"""
Simple test to check if bounding boxes fix the audio binding issue.
"""

import modal
from app_multitalk_cuda import app, generate_multi_person_video

@app.function()
def test_bbox_configurations():
    """Test different bbox configurations for multi-person video."""
    
    # Configuration 1: No bbox (current implementation)
    print("Test 1: No bounding boxes (current)")
    result1 = generate_multi_person_video.local(
        prompt="Two people having a conversation",
        image_key="multi1.png",
        audio_keys=["1.wav", "2.wav"],
        sample_steps=2,
        output_prefix="test_no_bbox"
    )
    
    return {
        "no_bbox": result1.get("success", False)
    }


if __name__ == "__main__":
    with app.run():
        results = test_bbox_configurations.remote()
        print(f"\nResults:")
        print(f"  No bbox: {results['no_bbox']}")