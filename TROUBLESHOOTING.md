# Troubleshooting Guide

## Common Issues and Solutions

### 1. ❌ CLIP Model Download Error

**Error Message:**
```
socket.gaierror: [Errno -2] Name or service not known
URLError: <urlopen error [Errno -2] Name or service not known>
```

**Cause:** No internet connection or network issue when trying to download CLIP model.

**Solutions:**

#### Option A: Connect to Internet and Download

1. **Ensure internet connection:**
   ```bash
   ping google.com  # Test connectivity
   ```

2. **Run the download script:**
   ```bash
   cd backend
   source venv/bin/activate
   python download_model.py
   ```

3. **Start the server:**
   ```bash
   python server.py
   ```

#### Option B: Manual Download

If automatic download fails:

1. **Download the model manually:**
   ```bash
   mkdir -p ~/.cache/clip
   
   # Download using wget
   wget https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt \
     -O ~/.cache/clip/ViT-B-32.pt
   
   # Or using curl
   curl -L https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt \
     -o ~/.cache/clip/ViT-B-32.pt
   ```

2. **Verify the download:**
   ```bash
   ls -lh ~/.cache/clip/ViT-B-32.pt
   # Should show ~338MB file
   ```

3. **Start the server:**
   ```bash
   cd backend
   python server.py
   ```

#### Option C: Use VPN

If download is blocked in your region:

1. Enable VPN
2. Run download script
3. Disable VPN (optional)
4. Run server

---

### 2. ❌ Port Already in Use

**Error Message:**
```
OSError: [Errno 48] Address already in use
```

**Solution:**

```bash
# Find process using port 5000
lsof -ti:5000

# Kill the process
lsof -ti:5000 | xargs kill -9

# Or use a different port
PORT=5001 python server.py
```

---

### 3. ❌ Module Not Found

**Error Message:**
```
ModuleNotFoundError: No module named 'flask'
```

**Solution:**

```bash
# Activate virtual environment
cd backend
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt

# Verify installation
pip list | grep -i flask
```

---

### 4. ❌ HDF5 File Not Found

**Error Message:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'images_embeds.h5'
```

**Solution:**

1. **Check if file exists:**
   ```bash
   ls -lh backend/images_embeds.h5
   ```

2. **If missing, you need to generate it:**
   - Run the Jupyter notebook `search_using_hnsw.ipynb`
   - Or copy from another location
   - Or generate from your image dataset

---

### 5. ❌ Frontend Can't Connect to Backend

**Error Message (in browser console):**
```
Failed to fetch
net::ERR_CONNECTION_REFUSED
```

**Solutions:**

1. **Verify backend is running:**
   ```bash
   curl http://localhost:5000/health
   ```

2. **Check environment variable:**
   ```bash
   cat client/.env.local
   # Should contain: NEXT_PUBLIC_API_URL=http://localhost:5000
   ```

3. **Check CORS settings:**
   - Backend should have `CORS(app)` enabled
   - Check browser console for CORS errors

4. **Restart both servers:**
   ```bash
   # Terminal 1
   ./start-backend.sh
   
   # Terminal 2
   ./start-frontend.sh
   ```

---

### 6. ❌ Images Not Loading

**Symptoms:** Blank cards or file paths displayed instead of images

**Solutions:**

1. **Check image paths in HDF5:**
   ```python
   import h5py
   with h5py.File("backend/images_embeds.h5", "r") as f:
       print(f["image_path"][:5])  # Show first 5 paths
   ```

2. **Verify images exist:**
   ```bash
   # Check if image directory exists
   ls -la backend/images/
   ```

3. **Check backend logs:**
   - Look for 404 or 500 errors
   - Verify image paths are correct

---

### 7. ❌ CUDA/GPU Issues

**Error Message:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Use CPU instead:**
   ```bash
   CUDA_VISIBLE_DEVICES="" python server.py
   ```

2. **Reduce batch size** (if processing multiple images)

3. **Check GPU availability:**
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
   ```

---

### 8. ❌ Build Errors (Frontend)

**Error Message:**
```
Module not found: Can't resolve 'react'
```

**Solution:**

```bash
cd client

# Clean install
rm -rf node_modules .next package-lock.json

# Reinstall
yarn install
# or: npm install

# Rebuild
yarn build
```

---

### 9. ❌ Vercel Deployment Issues

**Problem:** Frontend deployed but can't connect to backend

**Solution:**

1. **Set environment variable in Vercel:**
   ```bash
   vercel env add NEXT_PUBLIC_API_URL production
   # Enter your backend URL (e.g., https://your-app.railway.app)
   ```

2. **Redeploy:**
   ```bash
   vercel --prod
   ```

3. **Check backend CORS:**
   - Update `server.py` to allow your Vercel domain:
   ```python
   CORS(app, origins=["https://your-app.vercel.app"])
   ```

---

### 10. ❌ Search Returns No Results

**Symptoms:** Valid query but 0 results

**Possible Causes:**

1. **Empty or corrupted index:**
   - Verify HDF5 file has data
   - Check embeddings array size

2. **Query encoding failed:**
   - Check backend logs for errors
   - Verify CLIP model loaded correctly

3. **Threshold too high:**
   - HNSW returned distant matches
   - Try different query terms

---

## Quick Diagnostic Commands

```bash
# Check Python version
python --version  # Should be 3.8+

# Check Node version
node --version    # Should be 18+

# Check backend health
curl http://localhost:5000/health

# Test search endpoint
curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "k": 5}'

# Check frontend build
cd client && npm run build

# View backend logs
cd backend && python server.py 2>&1 | tee server.log

# Check disk space (model needs ~350MB)
df -h
```

## Getting Help

If issues persist:

1. **Check logs:**
   - Backend: `backend/server.log` or terminal output
   - Frontend: Browser DevTools Console
   - Vercel: Deployment logs in dashboard

2. **Verify setup:**
   - All dependencies installed
   - Virtual environment activated
   - Environment variables set
   - Ports not blocked by firewall

3. **Try clean install:**
   ```bash
   # Backend
   cd backend
   rm -rf venv
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   
   # Frontend
   cd client
   rm -rf node_modules .next
   yarn install
   ```

---

**Still having issues? Check the documentation:**
- [README.md](README.md) - Main overview
- [QUICKSTART.md](QUICKSTART.md) - Setup guide
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment instructions
- [ERROR_HANDLING.md](ERROR_HANDLING.md) - Image error handling 