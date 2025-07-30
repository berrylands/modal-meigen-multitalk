#!/bin/bash
#
# Test script for deploying and testing the REST API on Modal
#

set -e

echo "=========================================="
echo "MeiGen-MultiTalk REST API Deployment Test"
echo "=========================================="

# Check if Modal is installed
if ! command -v modal &> /dev/null; then
    echo "Error: Modal CLI not found. Please install with: pip install modal"
    exit 1
fi

# Check if logged in to Modal
if ! modal token whoami &> /dev/null; then
    echo "Error: Not logged in to Modal. Please run: modal token new"
    exit 1
fi

echo "✓ Modal CLI is installed and configured"

# Deploy the API in test mode
echo -e "\nDeploying REST API to Modal..."
modal deploy api.py --name multitalk-api-test

# Get the deployed URL
echo -e "\nGetting deployment URL..."
API_URL=$(modal app lookup multitalk-api-test | grep -oE 'https://[a-zA-Z0-9.-]+\.modal\.run' | head -1)

if [ -z "$API_URL" ]; then
    echo "Error: Could not get API URL from deployment"
    exit 1
fi

echo "✓ API deployed at: $API_URL"

# Wait for API to be ready
echo -e "\nWaiting for API to be ready..."
sleep 5

# Test health endpoint
echo -e "\nTesting health endpoint..."
HEALTH_RESPONSE=$(curl -s -w "\n%{http_code}" "$API_URL/api/v1/health")
HTTP_CODE=$(echo "$HEALTH_RESPONSE" | tail -1)
RESPONSE_BODY=$(echo "$HEALTH_RESPONSE" | head -n -1)

if [ "$HTTP_CODE" = "200" ]; then
    echo "✓ Health check passed"
    echo "  Response: $RESPONSE_BODY"
else
    echo "✗ Health check failed with HTTP $HTTP_CODE"
    echo "  Response: $RESPONSE_BODY"
    exit 1
fi

# Test authentication
echo -e "\nTesting authentication..."

# Test without API key (should fail)
AUTH_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/api/v1/generate" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"test"}')
HTTP_CODE=$(echo "$AUTH_RESPONSE" | tail -1)

if [ "$HTTP_CODE" = "403" ]; then
    echo "✓ Correctly rejected request without API key"
else
    echo "✗ Expected 403 without API key, got $HTTP_CODE"
fi

# Test with API key (in dev mode, any key works)
echo -e "\nTesting with API key..."
JOB_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/api/v1/generate" \
    -H "Authorization: Bearer test-api-key" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "Test deployment",
        "image_s3_url": "s3://test-bucket/test.png",
        "audio_s3_urls": ["s3://test-bucket/test.wav"]
    }')

HTTP_CODE=$(echo "$JOB_RESPONSE" | tail -1)
RESPONSE_BODY=$(echo "$JOB_RESPONSE" | head -n -1)

if [ "$HTTP_CODE" = "200" ]; then
    echo "✓ Job submission accepted"
    echo "  Response: $RESPONSE_BODY"
    
    # Extract job ID
    JOB_ID=$(echo "$RESPONSE_BODY" | grep -oE '"job_id":"[^"]+' | cut -d'"' -f4)
    
    if [ ! -z "$JOB_ID" ]; then
        echo "  Job ID: $JOB_ID"
        
        # Check job status
        echo -e "\nChecking job status..."
        sleep 2
        
        STATUS_RESPONSE=$(curl -s "$API_URL/api/v1/jobs/$JOB_ID" \
            -H "Authorization: Bearer test-api-key")
        
        echo "  Status response: $STATUS_RESPONSE"
    fi
else
    echo "✗ Job submission failed with HTTP $HTTP_CODE"
    echo "  Response: $RESPONSE_BODY"
fi

# Test API documentation
echo -e "\nChecking API documentation..."
DOCS_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/api/v1/docs")

if [ "$DOCS_CODE" = "200" ]; then
    echo "✓ API documentation available at: $API_URL/api/v1/docs"
else
    echo "✗ API documentation returned HTTP $DOCS_CODE"
fi

# Summary
echo -e "\n=========================================="
echo "Deployment Test Summary"
echo "=========================================="
echo "API URL: $API_URL"
echo "Docs URL: $API_URL/api/v1/docs"
echo ""
echo "To manually test the API:"
echo "  export MULTITALK_API_URL=$API_URL/api/v1"
echo "  export MULTITALK_API_KEY=your-api-key"
echo "  python test_api.py"
echo ""
echo "To remove test deployment:"
echo "  modal app stop multitalk-api-test"
echo "=========================================="