# Image Setup Guide

## ✅ Current Status

Your backend is **correctly configured** to serve images! Here's what's set up:

### Configuration Summary:
- **Images directory**: `backend/images/` ✅ (created)
- **Total images needed**: **498,349 images**
- **Image format**: `.jpg` files
- **Path structure**: Files should be named like `34894.jpg`, `59708.jpg`, etc.

## 📁 Where to Place Images

Place all your image files in:
```
/home/huy-pham/Workspace/dsa-advanced-assignment-hnsw/backend/images/
```

### Required Files:
Based on your HDF5 file, you need images named:
- `34894.jpg`
- `59708.jpg`
- `99930.jpg`
- ... (498,349 total files)

## 🔧 How the System Works

### 1. Image Paths in Database
The HDF5 file stores paths like:
```
./images/34894.jpg
./images/59708.jpg
```

### 2. Backend Resolution
The server automatically:
1. Removes the `./` prefix
2. Joins with backend directory: `/backend/images/34894.jpg`
3. Serves the image as base64 to frontend

### 3. Frontend Display
Frontend receives base64 image data and displays it.

## 🚀 Testing Setup

### Quick Test:
```bash
cd backend
conda run -n hnsw-backend-venv python test_image_paths.py
```

This will show:
- ✅ If images directory exists
- 📁 How many images are in the folder
- 📝 Sample paths and whether files exist

### Manual Test:
1. Add a test image (e.g., `34894.jpg`) to `backend/images/`
2. Start the server: `python server.py`
3. Test endpoint: `curl http://localhost:5000/image/./images/34894.jpg`
4. Should return base64 image data

## 📋 Step-by-Step: Adding Images

### Method 1: Copy All at Once
```bash
# If you have images in another folder
cp /path/to/your/images/*.jpg backend/images/

# Verify count
ls backend/images/ | wc -l
```

### Method 2: Download/Extract
```bash
# If images are in a zip file
unzip images.zip -d backend/images/

# If downloading from URL
wget -P backend/images/ http://example.com/images.tar.gz
tar -xzf backend/images/images.tar.gz -C backend/images/
```

### Method 3: Symbolic Link (if images are large)
```bash
# If images are elsewhere and you don't want to copy
ln -s /path/to/existing/images /home/huy-pham/Workspace/dsa-advanced-assignment-hnsw/backend/images
```

## ✅ Verification Checklist

Once you add images:

- [ ] Images are in `backend/images/` directory
- [ ] Files are named correctly (matching HDF5 paths)
- [ ] Run test script: `python test_image_paths.py`
- [ ] Start backend: `python server.py`
- [ ] Test API: `curl http://localhost:5000/image/./images/34894.jpg`
- [ ] Open frontend: Images should display!

## 🎯 Expected Behavior

### When Images Exist:
```
Frontend Search → Backend finds similar images → Returns paths like "./images/34894.jpg"
→ Backend serves image as base64 → Frontend displays image
```

### When Images Don't Exist:
```
Frontend Search → Backend finds similar images → Returns paths
→ Backend can't find file → Frontend shows placeholder with path
```

## 🔍 Troubleshooting

### Images not displaying?

1. **Check file exists:**
   ```bash
   ls -la backend/images/34894.jpg
   ```

2. **Check permissions:**
   ```bash
   chmod 644 backend/images/*.jpg
   ```

3. **Check server logs:**
   - Look for "Image not found" or "Cannot read image" errors

4. **Verify path in HDF5:**
   ```bash
   conda run -n hnsw-backend-venv python -c "import h5py; f=h5py.File('backend/images_embeds.h5','r'); print(f['image_path'][0])"
   ```

### Path mismatch?

The server expects:
- HDF5 path: `./images/34894.jpg`
- File location: `backend/images/34894.jpg`
- These should match!

## 📊 File Naming Examples

Your images should be named exactly as in the HDF5:

| HDF5 Path | Actual File Location |
|-----------|---------------------|
| `./images/34894.jpg` | `backend/images/34894.jpg` |
| `./images/59708.jpg` | `backend/images/59708.jpg` |
| `./images/99930.jpg` | `backend/images/99930.jpg` |

## 🎉 Ready to Go!

Once you place images in `backend/images/`:

1. ✅ Backend will serve them correctly
2. ✅ Frontend will display them in search results
3. ✅ Error handling will show paths if images missing

**The system is configured correctly - just add your image files!** 🚀

---

## Quick Commands Reference

```bash
# Check setup
cd backend && python test_image_paths.py

# Count images
ls backend/images/ | wc -l

# Test specific image
curl http://localhost:5000/image/./images/34894.jpg

# Start server
cd backend && python server.py

# Start frontend
cd client && npm run dev
``` 