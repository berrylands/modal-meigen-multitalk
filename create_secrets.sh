#!/bin/bash

# Load environment variables
source .env

# Export Modal token
export MODAL_AUTH_TOKEN=$MODAL_API_TOKEN

echo "Creating Modal secrets..."
echo ""

# Check if secrets already exist
echo "Checking existing secrets..."
modal secret list

echo ""
echo "Creating/updating secrets..."

# Create HuggingFace secret (if HUGGINGFACE_TOKEN is set)
if [ ! -z "$HUGGINGFACE_TOKEN" ]; then
    echo "Creating huggingface-secret..."
    modal secret create huggingface-secret HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN 2>/dev/null || \
    echo "  Secret already exists, skipping..."
else
    echo "  No HUGGINGFACE_TOKEN in .env, skipping..."
fi

# Create AWS secret (if AWS credentials are set)
if [ ! -z "$AWS_ACCESS_KEY_ID" ] && [ ! -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "Creating aws-secret..."
    modal secret create aws-secret \
        AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
        AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
        AWS_REGION=${AWS_REGION:-eu-west-1} 2>/dev/null || \
    echo "  Secret already exists, skipping..."
else
    echo "  No AWS credentials in .env, skipping..."
fi

echo ""
echo "Done! Current secrets:"
modal secret list