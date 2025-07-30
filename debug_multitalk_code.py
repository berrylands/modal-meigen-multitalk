#!/usr/bin/env python3
"""
Debug MultiTalk code to understand multi-person requirements.
"""

import modal
from app_multitalk_cuda import app, multitalk_cuda_image

@app.function(
    image=multitalk_cuda_image,
    volumes={"/models": modal.Volume.from_name("multitalk-models")},
)
def inspect_multitalk_code():
    """
    Inspect the MultiTalk code to understand multi-person audio handling.
    """
    import os
    
    print("Inspecting MultiTalk code for multi-person support...\n")
    
    # Look for audio_prepare_multi function
    generate_file = "/root/MultiTalk/generate_multitalk.py"
    
    if os.path.exists(generate_file):
        print("Found generate_multitalk.py, searching for audio_prepare_multi function...\n")
        
        with open(generate_file, 'r') as f:
            lines = f.readlines()
        
        # Find the audio_prepare_multi function
        in_function = False
        function_lines = []
        
        for i, line in enumerate(lines):
            if 'def audio_prepare_multi' in line:
                in_function = True
                print(f"Found audio_prepare_multi at line {i+1}:\n")
            
            if in_function:
                function_lines.append(f"{i+1}: {line.rstrip()}")
                
                # Look for audio_type usage
                if 'audio_type' in line:
                    print(f">>> AUDIO_TYPE usage: {line.strip()}")
                
                # Stop after finding the function body
                if len(function_lines) > 30:
                    break
        
        print("\nFunction definition:")
        print("\n".join(function_lines[:30]))
        
        # Also look for example JSON files
        print("\n\nSearching for example JSON files...")
        examples_dir = "/root/MultiTalk/examples"
        if os.path.exists(examples_dir):
            for file in os.listdir(examples_dir):
                if file.endswith('.json'):
                    print(f"\nFound example: {file}")
                    with open(os.path.join(examples_dir, file), 'r') as f:
                        import json
                        data = json.load(f)
                        print(json.dumps(data, indent=2))
        
        # Look for audio_type values
        print("\n\nSearching for audio_type values in code...")
        audio_type_lines = []
        for i, line in enumerate(lines):
            if 'audio_type' in line and ('==' in line or 'if' in line or 'elif' in line):
                audio_type_lines.append(f"Line {i+1}: {line.strip()}")
        
        if audio_type_lines:
            print("Found audio_type checks:")
            for line in audio_type_lines[:10]:
                print(f"  {line}")
    else:
        print(f"File not found: {generate_file}")
    
    return {"status": "inspection complete"}


if __name__ == "__main__":
    with app.run():
        result = inspect_multitalk_code.remote()
        print(f"\nResult: {result}")