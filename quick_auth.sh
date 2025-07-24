#!/bin/bash

echo "Modal Authentication Quick Start"
echo "================================"
echo ""
echo "This will open your browser for GitHub authentication."
echo "Please:"
echo "1. Click 'Authorize with GitHub'"
echo "2. Authorize Modal to access your GitHub account"
echo "3. Return to this terminal when complete"
echo ""
read -p "Press Enter to continue..."

# Run modal setup
modal setup

# Check if successful
if [ -f ~/.modal/config.json ]; then
    echo ""
    echo "✅ Authentication successful!"
    echo ""
    echo "Running verification..."
    python verify_setup.py
    echo ""
    echo "Testing connection..."
    python test_simple.py
else
    echo "❌ Authentication failed or was cancelled"
    exit 1
fi