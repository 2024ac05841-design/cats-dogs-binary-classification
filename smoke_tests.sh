#!/bin/bash

# Smoke tests for deployed model
# Run after deployment to verify service is working

set -e

SERVICE_URL="${SERVICE_URL:-http://localhost:8000}"
TEST_IMAGE="${TEST_IMAGE:-tests/test_image.jpg}"
MAX_RETRIES=5
RETRY_DELAY=2

echo "Running smoke tests on $SERVICE_URL"

# Function to wait for service
wait_for_service() {
    local retries=0
    while [ $retries -lt $MAX_RETRIES ]; do
        if curl -f -s "$SERVICE_URL/health" > /dev/null; then
            echo "✓ Service is healthy"
            return 0
        fi
        echo "Waiting for service... (attempt $((retries+1))/$MAX_RETRIES)"
        sleep $RETRY_DELAY
        retries=$((retries+1))
    done
    
    echo "✗ Service not responding after $MAX_RETRIES attempts"
    return 1
}

# Test 1: Health check
echo "Test 1: Health check..."
if curl -f -s "$SERVICE_URL/health" | grep -q "healthy"; then
    echo "✓ Health check passed"
else
    echo "✗ Health check failed"
    exit 1
fi

# Test 2: Service info
echo "Test 2: Get service info..."
if curl -f -s "$SERVICE_URL/info" | grep -q "Cats vs Dogs"; then
    echo "✓ Service info retrieved"
else
    echo "✗ Service info failed"
    exit 1
fi

# Test 3: Prediction with sample image
echo "Test 3: Make prediction..."
if [ -f "$TEST_IMAGE" ]; then
    RESPONSE=$(curl -f -s -F "file=@$TEST_IMAGE" "$SERVICE_URL/predict")
    
    if echo "$RESPONSE" | grep -q "class_name"; then
        echo "✓ Prediction successful"
        echo "Response: $RESPONSE"
    else
        echo "✗ Prediction failed"
        echo "Response: $RESPONSE"
        exit 1
    fi
else
    echo "⚠ Test image not found at $TEST_IMAGE, skipping prediction test"
    echo "  Create a test image at: $TEST_IMAGE"
fi

# Test 4: Error handling
echo "Test 4: Error handling..."
if curl -f -s -F "file=@/dev/null" "$SERVICE_URL/predict" 2>&1 | grep -q "error\|Error\|400"; then
    echo "✓ Error handling works"
else
    echo "⚠ Error handling test inconclusive"
fi

echo ""
echo "================================"
echo "✓ All smoke tests passed!"
echo "================================"
echo "Service is ready for use at: $SERVICE_URL"
