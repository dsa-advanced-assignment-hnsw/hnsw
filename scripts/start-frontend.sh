#!/bin/bash

echo "🚀 Starting HNSW Image Search Frontend..."

cd client

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    yarn install || npm install
fi

# Check if .env.local exists
if [ ! -f ".env.local" ]; then
    echo "🔧 Creating .env.local file..."
    echo "NEXT_PUBLIC_API_URL=http://localhost:5000" > .env.local
fi

# Run the development server
echo "✅ Starting Next.js development server on http://localhost:3000"
yarn dev || npm run dev 