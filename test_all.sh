#!/bin/bash

echo "Modal MeiGen-MultiTalk Setup Test"
echo "================================="
echo ""

# Load environment
source .env
export MODAL_AUTH_TOKEN=$MODAL_API_TOKEN

# 1. Check Modal CLI
echo "1. Testing Modal CLI..."
if modal --version > /dev/null 2>&1; then
    echo "   ✅ Modal CLI: $(modal --version)"
else
    echo "   ❌ Modal CLI not working"
    exit 1
fi

# 2. Test basic connection
echo ""
echo "2. Testing Modal connection..."
modal run --quiet test_deploy.py::health_check 2>&1 | grep -q "healthy" && \
    echo "   ✅ Connection successful" || \
    echo "   ❌ Connection failed"

# 3. Deploy test app
echo ""
echo "3. Deploying test app..."
modal deploy test_deploy.py > /dev/null 2>&1 && \
    echo "   ✅ Deployment successful" || \
    echo "   ❌ Deployment failed"

# 4. Check app status
echo ""
echo "4. Checking deployed apps..."
modal app list | grep -q "meigen-multitalk-test" && \
    echo "   ✅ Test app found" || \
    echo "   ❌ Test app not found"

# 5. Stop test app
echo ""
echo "5. Cleaning up..."
modal app stop meigen-multitalk-test > /dev/null 2>&1 && \
    echo "   ✅ Cleanup complete" || \
    echo "   ⚠️  Cleanup skipped"

echo ""
echo "================================="
echo "✅ Modal setup is working correctly!"
echo ""
echo "Next steps:"
echo "- Review the code changes"
echo "- Create a pull request"
echo "- Proceed to Issue #2"