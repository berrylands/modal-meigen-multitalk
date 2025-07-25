"""
Test the final production image with exact Colab configuration.
"""

import modal
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
modal.enable_output()

if "MODAL_API_TOKEN" in os.environ:
    os.environ["MODAL_AUTH_TOKEN"] = os.environ["MODAL_API_TOKEN"]

from modal_image_production_final import multitalk_image, multitalk_image_no_flash

# Test which version - change to test different options
TEST_FLASH = False  # Set to False to test without flash-attn

if TEST_FLASH:
    test_image = multitalk_image
    print("Testing with flash-attn 2.6.1 (Colab configuration)")
else:
    test_image = multitalk_image_no_flash
    print("Testing without flash-attn (xformers only)")

app = modal.App("test-final-production")

@app.function(
    image=test_image,
    gpu="a100" if TEST_FLASH else "a10g",  # A100 for flash-attn as in Colab
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("aws-secret"),
    ],
    timeout=600,
)
def test_production():
    """Test the production environment."""
    import torch
    import sys
    import os
    
    print("="*60)
    print("MeiGen-MultiTalk Production Test (Colab Configuration)")
    print("="*60)
    
    # System info
    print(f"\nSystem:")
    print(f"Python: {sys.version.split()[0]}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Test critical packages with versions
    print("\nPackage Versions (matching Colab):")
    packages = {
        "torch": "2.4.1",
        "transformers": "4.49.0", 
        "xformers": "0.0.28",
    }
    
    all_match = True
    for pkg, expected in packages.items():
        try:
            module = __import__(pkg)
            actual = module.__version__
            match = actual == expected
            all_match &= match
            print(f"  {pkg}: {actual} {'✅' if match else f'❌ (expected {expected})'}")
        except ImportError:
            print(f"  {pkg}: ❌ NOT INSTALLED")
            all_match = False
    
    # Test flash-attn if included
    if TEST_FLASH:
        print("\nTesting flash-attn 2.6.1:")
        try:
            import flash_attn
            print(f"  Version: {flash_attn.__version__}")
            
            # Test functionality
            from flash_attn import flash_attn_func
            b, s, h, d = 2, 128, 8, 64
            q = torch.randn(b, s, h, d).cuda().half()
            k = torch.randn(b, s, h, d).cuda().half() 
            v = torch.randn(b, s, h, d).cuda().half()
            out = flash_attn_func(q, k, v)
            print(f"  ✅ Flash attention working! Output: {out.shape}")
        except ImportError:
            print("  ❌ flash-attn not installed")
            all_match = False
        except Exception as e:
            print(f"  ❌ flash-attn error: {str(e)[:100]}")
            all_match = False
    else:
        print("\nTesting xformers memory efficient attention:")
        try:
            import xformers.ops
            b, h, s, d = 2, 8, 128, 64
            q = torch.randn(b, s, h, d).cuda()
            k = torch.randn(b, s, h, d).cuda()
            v = torch.randn(b, s, h, d).cuda()
            out = xformers.ops.memory_efficient_attention(q, k, v)
            print(f"  ✅ xformers attention working! Output: {out.shape}")
        except Exception as e:
            print(f"  ❌ xformers error: {str(e)[:100]}")
            all_match = False
    
    # Check MultiTalk
    print("\nMultiTalk Repository:")
    multitalk_exists = os.path.exists("/root/MultiTalk")
    print(f"  Repository: {'✅ Found' if multitalk_exists else '❌ Not found'}")
    all_match &= multitalk_exists
    
    # Check secrets
    print("\nSecrets:")
    hf_ok = any(k in os.environ for k in ["HF_TOKEN", "HUGGINGFACE_TOKEN"])
    aws_ok = "AWS_ACCESS_KEY_ID" in os.environ
    print(f"  HuggingFace: {'✅' if hf_ok else '❌'}")
    print(f"  AWS: {'✅' if aws_ok else '❌'}")
    all_match &= hf_ok and aws_ok
    
    print("\n" + "="*60)
    if all_match:
        print("✅ ALL CHECKS PASSED - Matches Colab configuration!")
    else:
        print("❌ Some checks failed - see above")
    print("="*60)
    
    return {"success": all_match}

if __name__ == "__main__":
    print("\nBuilding and testing production image...")
    print("This may take several minutes if flash-attn needs compilation...\n")
    
    try:
        with app.run():
            result = test_production.remote()
            
            if result["success"]:
                print("\n✅ Production image is ready!")
                print("All components match the Colab implementation.")
            else:
                print("\n❌ Production image has issues.")
                print("Check the output above for details.")
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {str(e)[:200]}")
        print("\nThis might be a build timeout. Flash-attn compilation can take 5-10 minutes.")