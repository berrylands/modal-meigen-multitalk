#!/usr/bin/env python3
"""
Debug audio-character binding issue.
"""

# ARCHIVED - DO NOT USE
# import modal
# ARCHIVED - DO NOT USE
# from app_multitalk_cuda import app, multitalk_cuda_image, model_volume, hf_cache_volume

# @app.function(
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
    timeout=1200,
)
def debug_audio_binding():
    """Debug why both characters use the same audio."""
    import os
    import json
    import subprocess
    
    os.chdir("/root/MultiTalk")
    
    # Check the MultiTalk code for audio handling
    print("Checking MultiTalk audio handling code...\n")
    
    # Look for the audio_prepare_multi function
    with open("generate_multitalk.py", "r") as f:
        content = f.read()
    
    # Find relevant sections
    if "audio_prepare_multi" in content:
        print("Found audio_prepare_multi function")
        # Extract the function
        start = content.find("def audio_prepare_multi")
        if start != -1:
            end = content.find("\ndef ", start + 1)
            if end == -1:
                end = start + 1000  # Get next 1000 chars
            func_content = content[start:end]
            print("\nFunction snippet:")
            print(func_content[:500])
    
    # Check for L-RoPE references
    lrope_refs = []
    for i, line in enumerate(content.split('\n')):
        if 'rope' in line.lower() or 'lrope' in line.lower() or 'l-rope' in line.lower():
            lrope_refs.append(f"Line {i+1}: {line.strip()}")
    
    if lrope_refs:
        print(f"\nFound {len(lrope_refs)} L-RoPE references:")
        for ref in lrope_refs[:5]:
            print(f"  {ref}")
    
    # Check for audio_type handling
    audio_type_refs = []
    for i, line in enumerate(content.split('\n')):
        if 'audio_type' in line and ('para' in line or 'if' in line):
            audio_type_refs.append(f"Line {i+1}: {line.strip()}")
    
    if audio_type_refs:
        print(f"\nFound audio_type handling:")
        for ref in audio_type_refs[:5]:
            print(f"  {ref}")
    
    # Check example JSONs
    print("\n\nChecking example JSONs...")
    examples_dir = "examples"
    if os.path.exists(examples_dir):
        # Look for multi-person examples
        for root, dirs, files in os.walk(examples_dir):
            for file in files:
                if file.endswith('.json') and 'multi' in file:
                    json_path = os.path.join(root, file)
                    print(f"\nFound: {json_path}")
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        print(json.dumps(data, indent=2))
    
    # Test with different audio configurations
    print("\n\nTesting different audio configurations...")
    
    # Download test files
    import boto3
    bucket_name = os.environ.get('AWS_BUCKET_NAME')
    s3 = boto3.client('s3')
    
    s3.download_file(bucket_name, "multi1.png", "test_image.png")
    s3.download_file(bucket_name, "1.wav", "audio1.wav")
    s3.download_file(bucket_name, "2.wav", "audio2.wav")
    
    # Test 1: With explicit audio_type
    test_configs = [
        {
            "name": "para_mode",
            "json": {
                "prompt": "Two people having a conversation",
                "cond_image": "test_image.png",
                "audio_type": "para",
                "cond_audio": {
                    "person1": "audio1.wav",
                    "person2": "audio2.wav"
                }
            }
        },
        {
            "name": "concat_mode",
            "json": {
                "prompt": "Two people having a conversation", 
                "cond_image": "test_image.png",
                "audio_type": "concat",
                "cond_audio": {
                    "person1": "audio1.wav",
                    "person2": "audio2.wav"
                }
            }
        }
    ]
    
    results = []
    for config in test_configs:
        print(f"\n\nTesting {config['name']}...")
        with open(f"test_{config['name']}.json", "w") as f:
            json.dump(config['json'], f)
        
        # Run minimal test
        cmd = [
            "python3", "generate_multitalk.py",
            "--ckpt_dir", "weights/Wan2.1-I2V-14B-480P",
            "--wav2vec_dir", "weights/chinese-wav2vec2-base",
            "--input_json", f"test_{config['name']}.json",
            "--frame_num", "21",
            "--sample_steps", "2",
            "--save_file", f"output_{config['name']}",
        ]
        
        # Just check if it starts correctly
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        results.append({
            "config": config['name'],
            "returncode": result.returncode,
            "error": result.stderr[:200] if result.stderr else "No error"
        })
    
    return {
        "lrope_found": len(lrope_refs) > 0,
        "audio_type_handling": len(audio_type_refs) > 0,
        "test_results": results
    }


if __name__ == "__main__":
    with app.run():
        result = debug_audio_binding.remote()
        print(f"\nDebug results:")
        print(f"  L-RoPE found: {result.get('lrope_found')}")
        print(f"  Audio type handling: {result.get('audio_type_handling')}")
        print(f"\nTest results:")
        for test in result.get('test_results', []):
            print(f"  {test['config']}: code {test['returncode']}")
            if test['returncode'] != 0:
                print(f"    Error: {test['error']}")