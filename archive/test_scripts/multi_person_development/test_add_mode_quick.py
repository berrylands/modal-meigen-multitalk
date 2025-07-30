#!/usr/bin/env python3
"""
Quick test of 'add' mode with minimal frames.
"""

import modal
from app_multitalk_cuda import app, generate_multi_person_video

@app.function()
def test_add_mode_quick():
    """Quick test with 'add' mode for sequential speaking."""
    
    print("Testing 'add' mode with minimal frames for quick iteration...\n")
    
    # Force 45 frames for faster generation
    result = generate_multi_person_video.local(
        prompt="Two people having a conversation, taking turns to speak",
        image_key="multi1.png",
        audio_keys=["1.wav", "2.wav"],
        sample_steps=10,  # Min for good lip sync
        output_prefix="test_add_quick",
        audio_type="add",  # Sequential speaking
        use_bbox=False,
        audio_cfg=4.0
    )
    
    return result


if __name__ == "__main__":
    with app.run():
        result = test_add_mode_quick.remote()
        if result.get("success"):
            print(f"\n✅ Success!")
            print(f"S3 output: {result.get('s3_output')}")
            print(f"Frame count: {result.get('frame_count')}")
            print(f"Mode: Sequential (add)")
        else:
            print(f"\n❌ Failed: {result.get('error', 'Unknown')}")