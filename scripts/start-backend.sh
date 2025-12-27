#!/bin/bash

echo "🚀 Starting HNSW Image Search Backend..."

cd backend

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
echo "📥 Installing dependencies..."
pip install -r requirements.txt -q

# Check if images_embeds.h5 exists
if [ ! -f "images_embeds.h5" ]; then
    echo "⚠️  Warning: images_embeds.h5 not found!"
    echo "Please make sure the HDF5 file is in the backend directory."
    exit 1
fi

# Run the server
echo "✅ Starting Flask server on http://localhost:5000"
echo "⏳ Loading CLIP model and HNSW index... this may take 1-2 minutes"
echo ""

python server.py 