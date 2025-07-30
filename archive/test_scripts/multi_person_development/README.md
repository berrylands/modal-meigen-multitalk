# Multi-Person Development Test Scripts

This directory contains test scripts created during the development of multi-person conversation support for MeiGen-MultiTalk.

## Scripts

- `debug_audio_binding.py` - Debugged audio-character binding issues
- `test_add_mode_quick.py` - Quick test for sequential (add) audio mode
- `test_audio_binding.py` - Compared single vs multi-person audio handling
- `test_audio_modes.py` - Tested different audio modes (add vs para)
- `test_bbox_simple.py` - Simple bounding box configuration tests
- `test_with_bbox.py` - Comprehensive bounding box testing
- `verify_inputs.py` - Verified JSON input structure matches official examples

## Key Learnings

1. **Audio Modes**: "add" mode for sequential speaking, "para" for simultaneous
2. **Frame Count**: Let the model use default (81) instead of calculating
3. **Bounding Boxes**: Not required for basic multi-person generation
4. **Color Correction**: Helps prevent brightness issues on characters

These scripts were instrumental in understanding and fixing:
- Audio-character binding issues
- Simultaneous vs sequential speaking
- Brightness/clarity problems
- Frame count calculations

Archived on: 2025-07-30