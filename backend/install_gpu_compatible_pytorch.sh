#!/bin/bash

echo "🔧 Installing PyTorch compatible with CUDA 5.0 (for NVIDIA GeForce MX130)"
echo ""
echo "Your GPU: NVIDIA GeForce MX130 (CUDA Capability 5.0)"
echo "Current PyTorch: 2.8.0 (requires CUDA 7.0+)"
echo "Installing: PyTorch 1.9.0 + CUDA 10.2 (supports CUDA 5.0)"
echo ""

# First, uninstall current PyTorch
echo "📦 Uninstalling current PyTorch..."
pip uninstall -y torch torchvision

# Install compatible PyTorch version
echo "📥 Installing PyTorch 1.9.0 with CUDA 10.2 support..."
pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html

# Verify installation
echo ""
echo "✅ Installation complete!"
echo ""
echo "🔍 Verifying GPU compatibility..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}' if torch.cuda.is_available() else 'CPU only'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU detected')"

echo ""
echo "🎉 Done! Your GPU should now work with PyTorch." 