#!/bin/bash
# Install flash-attn in Modal environment

set -e

echo "Installing flash-attn dependencies..."
apt-get update
apt-get install -y build-essential ninja-build

echo "Installing Python dependencies..."
pip install ninja packaging

echo "Installing flash-attn..."
pip install flash-attn==2.6.1 --no-build-isolation

echo "Flash-attn installation complete!"