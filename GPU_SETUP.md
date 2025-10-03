# GPU Setup Guide

## Using Your NVIDIA GPU with PyTorch

### Current Situation

Your **NVIDIA GeForce MX130** has:
- CUDA Capability: **5.0** (sm_50)
- Architecture: Maxwell

However, modern PyTorch versions (2.0+) require:
- Minimum CUDA Capability: **7.0** (sm_70)
- Supported: Volta, Turing, Ampere, Hopper, etc.

**Solution:** Install an older PyTorch version that supports CUDA 5.0

---

## Quick GPU Setup (Recommended)

### Step 1: Stop the Server

```bash
# Find and stop the running server
pkill -f "python server.py"
```

### Step 2: Install Compatible PyTorch

```bash
cd backend
conda activate hnsw-backend-venv  # or: source venv/bin/activate

# Run the installation script
./install_gpu_compatible_pytorch.sh
```

This will:
1. ✅ Uninstall PyTorch 2.8.0
2. ✅ Install PyTorch 1.9.0 + CUDA 10.2 (supports CUDA 5.0)
3. ✅ Verify GPU detection

### Step 3: Restart Server

```bash
python server.py
```

You should see:
```
✅ Loaded ... images into search index
```

And the device will show as `cuda` instead of `cpu`!

---

## Manual Installation (Alternative)

If the script doesn't work, install manually:

```bash
# Activate environment
conda activate hnsw-backend-venv

# Uninstall current PyTorch
pip uninstall -y torch torchvision

# Install PyTorch 1.9.0 with CUDA 10.2
pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
```

---

## Verify GPU is Working

```bash
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
"
```

**Expected Output:**
```
PyTorch version: 1.9.0+cu102
CUDA available: True
CUDA version: 10.2
GPU: NVIDIA GeForce MX130
GPU Memory: 2.00 GB
```

---

## Performance Comparison

### CPU Mode (Current)
- Text encoding: ~500ms per query
- Image encoding: ~1-2s per image

### GPU Mode (After upgrade)
- Text encoding: ~50-100ms per query ⚡ **5-10x faster**
- Image encoding: ~200-400ms per image ⚡ **3-5x faster**

---

## Supported PyTorch Versions for CUDA 5.0

| PyTorch Version | CUDA Version | Supports MX130 | Recommended |
|----------------|--------------|----------------|-------------|
| 1.7.1 | 10.1 | ✅ Yes | Good |
| 1.9.0 | 10.2 | ✅ Yes | ⭐ **Best** |
| 1.10.0 | 10.2 | ✅ Yes | Good |
| 2.0+ | 11.7+ | ❌ No | - |

---

## Troubleshooting

### Issue: "CUDA out of memory"

Your GPU has 2GB memory. If you get OOM errors:

```bash
# Reduce batch size or use CPU for this operation
export CUDA_VISIBLE_DEVICES=""
python server.py
```

### Issue: "CUDA driver version is insufficient"

Update your NVIDIA drivers:

```bash
# Check current driver
nvidia-smi

# Update on Ubuntu/Debian
sudo apt update
sudo apt install nvidia-driver-470

# Reboot
sudo reboot
```

### Issue: Still shows CPU

1. **Check CUDA installation:**
   ```bash
   nvcc --version
   nvidia-smi
   ```

2. **Verify PyTorch sees GPU:**
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

3. **Check CLIP model loads on GPU:**
   ```python
   import clip
   import torch
   device = "cuda" if torch.cuda.is_available() else "cpu"
   model, preprocess = clip.load("ViT-B/32", device=device)
   print(f"Model on: {next(model.parameters()).device}")
   ```

---

## Alternative: Use CPU Mode

If GPU setup is too complex, CPU mode works fine:

**Pros:**
- ✅ No compatibility issues
- ✅ Stable and reliable
- ✅ Works everywhere

**Cons:**
- ⏱️ 5-10x slower than GPU
- ⏱️ Higher latency for searches

To force CPU mode:
```bash
export CUDA_VISIBLE_DEVICES=""
python server.py
```

---

## Updating Requirements

After installing PyTorch 1.9.0, update your requirements:

```bash
# Freeze current environment
pip freeze | grep torch > torch_requirements.txt

# Content should be:
# torch==1.9.0+cu102
# torchvision==0.10.0+cu102
```

---

## Server Startup with GPU

After installation, when you run:

```bash
python server.py
```

You should see:
```
✅ Loaded 23 images into search index
Device: cuda  ← GPU is active!
```

Instead of:
```
Device: cpu  ← CPU mode
```

---

## Questions?

- **Why not use latest PyTorch?** → Your GPU is too old (CUDA 5.0 vs required 7.0+)
- **Is PyTorch 1.9.0 safe?** → Yes, stable and widely used. CLIP works perfectly.
- **Will CLIP work?** → Yes! CLIP is compatible with PyTorch 1.9.0
- **What about other GPUs?** → Newer GPUs (GTX 10-series+) use latest PyTorch

---

**Ready to enable your GPU? Run the installation script!** 🚀

```bash
cd backend
./install_gpu_compatible_pytorch.sh
``` 