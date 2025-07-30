#!/usr/bin/env python3
"""
Verify our multi-person inputs match what MultiTalk expects.
"""

import json

# What we're creating
our_json = {
    "prompt": "Two people having an animated conversation",
    "cond_image": "input.png",
    "audio_type": "para",
    "cond_audio": {
        "person1": "input_person1.wav",
        "person2": "input_person2.wav"
    }
}

# What the official example shows
official_json = {
    "prompt": "In a cozy recording studio, a man and a woman are singing together...",
    "cond_image": "examples/multi/2/multi2.png",
    "audio_type": "para",
    "cond_audio": {
        "person1": "examples/multi/2/1.wav",
        "person2": "examples/multi/2/1.wav"
    }
}

print("Our JSON structure:")
print(json.dumps(our_json, indent=2))

print("\nOfficial example structure:")
print(json.dumps(official_json, indent=2))

print("\nKey differences:")
print("1. Prompt length:", len(our_json["prompt"]), "vs", len(official_json["prompt"]))
print("2. Audio files same?", our_json["cond_audio"]["person1"] == our_json["cond_audio"]["person2"], 
      "vs", official_json["cond_audio"]["person1"] == official_json["cond_audio"]["person2"])

# Check if there are any other parameters we might be missing
print("\nChecking for missing parameters...")
for key in official_json:
    if key not in our_json:
        print(f"Missing: {key}")

print("\nStructure matches: ✅" if set(our_json.keys()) == set(official_json.keys()) else "Structure differs: ❌")