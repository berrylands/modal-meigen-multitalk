#!/usr/bin/env python3
"""
Final debug to understand the exact issue.
"""

import modal
import os

modal.enable_output()

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

debug_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["git", "ffmpeg"])
    .pip_install("boto3", "Pillow", "numpy")
)

app = modal.App("debug-final")

@app.function(
    image=debug_image,
    secrets=[modal.Secret.from_name("aws-secret")],
    timeout=300,
)
def check_image_details():
    """
    Check exact details of the input image.
    """
    import boto3
    from PIL import Image
    import numpy as np
    
    print("Checking input image details...")
    
    bucket_name = os.environ.get('AWS_BUCKET_NAME')
    s3 = boto3.client('s3')
    
    # Download image
    s3.download_file(bucket_name, "multi1.png", "/tmp/multi1.png")
    
    # Analyze image
    img = Image.open("/tmp/multi1.png")
    
    print(f"\nImage Details:")
    print(f"  Size: {img.size} (width x height)")
    print(f"  Mode: {img.mode}")
    print(f"  Format: {img.format}")
    
    # Check if dimensions work with VAE
    width, height = img.size
    
    print(f"\nVAE Analysis (8x downsampling):")
    print(f"  Latent width: {width} / 8 = {width / 8}")
    print(f"  Latent height: {height} / 8 = {height / 8}")
    
    # The error shape [1, 11, 4, 56, 112] suggests expected latent of 56x112
    expected_width = 112 * 8  # 896
    expected_height = 56 * 8   # 448
    
    print(f"\nExpected dimensions based on error:")
    print(f"  Expected: {expected_width}x{expected_height} (896x448)")
    print(f"  Actual: {width}x{height}")
    print(f"  Match: {width == expected_width and height == expected_height}")
    
    # 480p standard resolutions
    print(f"\n480p standard resolutions:")
    resolutions = [
        (854, 480),  # 16:9
        (640, 480),  # 4:3
        (896, 448),  # Possibly what MultiTalk expects?
        (448, 896),  # Portrait version?
    ]
    
    for w, h in resolutions:
        latent_w = w / 8
        latent_h = h / 8
        print(f"  {w}x{h} -> latent {latent_w}x{latent_h}")
    
    # Test resizing
    print(f"\n🔄 Testing resize to 896x448...")
    resized = img.resize((896, 448), Image.Resampling.LANCZOS)
    resized.save("/tmp/resized.png")
    print(f"  Resized and saved")
    
    return {
        "original_size": img.size,
        "needs_resize": not (width == expected_width and height == expected_height),
        "expected_size": (expected_width, expected_height)
    }


if __name__ == "__main__":
    with app.run():
        print("Debugging image requirements...\n")
        
        result = check_image_details.remote()
        
        print(f"\nResult: {result}")
        
        if result.get("needs_resize"):
            print(f"\n⚠️  Image needs to be resized!")
            print(f"  Current: {result['original_size']}")
            print(f"  Expected: {result['expected_size']}")
            print(f"\nThis explains the shape mismatch error!")
