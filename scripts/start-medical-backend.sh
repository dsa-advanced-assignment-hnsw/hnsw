#!/bin/bash

# Start Medical Search Backend Server
# Runs on port 5002 by default

echo "🏥 Starting Medical Search Backend (Bone Fractures)"
echo "=================================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda not found!"
    echo "   Please install Miniconda or Anaconda first."
    exit 1
fi

# Check if environment exists
if ! conda env list | grep -q "hnsw-backend-venv"; then
    echo "❌ Error: conda environment 'hnsw-backend-venv' not found!"
    echo ""
    echo "Create it with:"
    echo "  conda create -n hnsw-backend-venv python=3.10"
    echo "  conda activate hnsw-backend-venv"
    echo "  cd backend && pip install -r requirements-clean.txt"
    exit 1
fi

# Check if HDF5 file exists
if [ ! -f "backend/Medical_Fractures_Embedbed.h5" ]; then
    echo "⚠️  Warning: Medical_Fractures_Embedbed.h5 not found in backend/"
    echo ""
    echo "Please generate embeddings first:"
    echo "  cd medical_embedder"
    echo "  python generate_embeddings.py --cloud-name YOUR_CLOUD_NAME --api-key YOUR_API_KEY --api-secret YOUR_API_SECRET"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "📂 Working directory: $(pwd)"
echo "🐍 Activating conda environment: hnsw-backend-venv"
echo ""

# Activate conda and run server
cd backend
eval "$(conda shell.bash hook)"
conda activate hnsw-backend-venv

echo "🚀 Starting server on http://localhost:5002"
echo "   Press Ctrl+C to stop"
echo ""

python server_medical.py

