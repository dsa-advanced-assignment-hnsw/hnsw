#!/bin/bash

# Test Medical Backend API

echo "Testing Medical Backend API"
echo "============================"
echo ""

# Test 1: Health Check
echo "1. Testing health endpoint..."
curl -s http://localhost:5002/health | jq '.'
echo ""

# Test 2: Text Search
echo "2. Testing text search..."
curl -s -X POST http://localhost:5002/search \
  -H "Content-Type: application/json" \
  -d '{"query": "broken leg", "k": 3}' | jq '.results[0]'
echo ""

# Test 3: Get first image path
echo "3. Getting image path from search results..."
IMAGE_PATH=$(curl -s -X POST http://localhost:5002/search \
  -H "Content-Type: application/json" \
  -d '{"query": "broken leg", "k": 1}' | jq -r '.results[0].path')

echo "Image path: $IMAGE_PATH"
echo ""

# Test 4: Try to fetch the image
if [ ! -z "$IMAGE_PATH" ]; then
  echo "4. Testing image fetch..."
  ENCODED_PATH=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$IMAGE_PATH', safe=''))")
  echo "Encoded path: $ENCODED_PATH"
  echo ""
  
  curl -s "http://localhost:5002/image/$ENCODED_PATH" | jq -r '.image_data' | head -c 100
  echo "..."
  echo ""
  echo "✅ If you see 'data:image' above, the image endpoint is working!"
else
  echo "❌ No image path found in search results"
fi

echo ""
echo "Testing complete!"


