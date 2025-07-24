# Test Suite

This directory contains the essential tests for the Modal MeiGen-MultiTalk project.

## Available Tests

### 1. `verify_setup.py`
Quick verification of Modal installation and configuration status.
```bash
python verify_setup.py
```

### 2. `test_simple.py`
Basic Modal connectivity test. Verifies authentication and basic function execution.
```bash
python test_simple.py
```

### 3. `test_complete_setup.py`
Comprehensive test that verifies:
- AWS S3 access and credentials
- HuggingFace authentication
- Modal deployment capabilities
```bash
modal run test_complete_setup.py::verify_complete_setup
```

## Helper Scripts

### `create_secrets.sh`
Utility script to create or verify Modal secrets for AWS and HuggingFace.
```bash
./create_secrets.sh
```

## Running Tests

1. **Quick Check**: Start with `verify_setup.py`
2. **Basic Test**: Run `test_simple.py` 
3. **Full Verification**: Use `test_complete_setup.py`

All tests should pass before proceeding with model deployment.