#!/usr/bin/env python3
"""
Test single-person video using the multi-person function as a workaround.
Since multi-person works perfectly, we can use it with a single audio file.
"""

import modal

# Import the app
app = modal.App("test-single-workaround")

# Import the multi-person function from the deployed app
@app.function()
def test_single_as_multi():
    # Import and call the multi-person function directly
    from app_multitalk_cuda import generate_multi_person_video
    
    print("Testing single-person video using multi-person function...")
    print("This should work since multi-person function handles audio correctly")
    
    result = generate_multi_person_video.local(
        prompt="A person speaking naturally with clear lip sync",
        image_key="multi1.png",
        audio_keys=["1.wav"],  # Single audio file as a list
        sample_steps=20,
        output_prefix="single_workaround"
    )
    
    return result

if __name__ == "__main__":
    with app.run():
        result = test_single_as_multi.remote()
        print(f"Result: {result}")