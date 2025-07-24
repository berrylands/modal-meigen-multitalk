#!/bin/bash

# Load environment variables
source .env

# Export Modal token
export MODAL_AUTH_TOKEN=$MODAL_API_TOKEN

# Run the test
echo "Testing Modal connection with API token..."
python test_simple.py