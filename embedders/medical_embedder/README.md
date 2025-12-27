# Medical Image Embedder - Bone Fracture X-rays

Generate embeddings for bone fracture X-ray images using BiomedCLIP and host them on Cloudinary CDN for the medical search engine.

## Overview

This directory contains tools to:
1. Download the FracAtlas bone fracture dataset
2. Upload X-ray images to Cloudinary CDN
3. Generate 512-dimensional embeddings using BiomedCLIP
4. Save embeddings and URLs to HDF5 format
5. Create HNSW index for fast similarity search

## Prerequisites

### 1. Conda Environment

Activate the backend conda environment:

```bash
conda activate hnsw-backend-venv
```

If the environment doesn't exist:

```bash
conda create -n hnsw-backend-venv python=3.10
conda activate hnsw-backend-venv
```

### 2. Cloudinary Account

Create a free Cloudinary account:
- Visit: https://cloudinary.com/users/register/free
- Free tier: 25GB storage, 25GB bandwidth/month
- See `CLOUDINARY_SETUP.md` for detailed setup instructions

### 3. Disk Space

- ~500MB for FracAtlas dataset
- ~1GB temporary space during processing
- ~50MB for final HDF5 file

### 4. GPU (Optional but Recommended)

- BiomedCLIP runs on CPU but is much faster on GPU
- CUDA-compatible GPU recommended for large datasets

## Quick Start

### Step 1: Install Dependencies

```bash
conda activate hnsw-backend-venv
cd medical_embedder
pip install transformers huggingface-hub cloudinary pillow requests h5py tqdm numpy
```

Or install from requirements:

```bash
cd ../backend
pip install -r requirements-clean.txt
```

### Step 2: Set Up Cloudinary

1. Create account at https://cloudinary.com/users/register/free
2. Get your credentials from Dashboard:
   - Cloud Name
   - API Key
   - API Secret
3. See `CLOUDINARY_SETUP.md` for detailed instructions

### Step 3: Run the Notebook

```bash
# Make sure you're in the conda environment
conda activate hnsw-backend-venv

# Start Jupyter
jupyter notebook medical_embedder.ipynb
```

Or use JupyterLab:

```bash
jupyter lab medical_embedder.ipynb
```

### Step 4: Configure Cloudinary in Notebook

In Cell 2 of the notebook, replace with your credentials:

```python
cloudinary.config(
    cloud_name="your_cloud_name",    # From Dashboard
    api_key="your_api_key",          # From Dashboard
    api_secret="your_api_secret"     # From Dashboard
)
```

### Step 5: Run All Cells

Execute all cells in order:
1. Install dependencies
2. Configure Cloudinary
3. Download FracAtlas dataset
4. Upload images to Cloudinary
5. Load BiomedCLIP model
6. Generate embeddings
7. Save to HDF5
8. Verify output

**Time estimate**: 1-2 hours for full pipeline (4,083 images)

### Step 6: Copy HDF5 to Backend

```bash
cp Medical_Fractures_Embedbed.h5 ../backend/
```

### Step 7: Start Medical Search Server

```bash
cd ../backend
conda activate hnsw-backend-venv
python server_medical.py
```

Server will start on http://localhost:5002

## Files in This Directory

```
medical_embedder/
├── README.md                    # This file
├── CLOUDINARY_SETUP.md         # Cloudinary setup guide
├── DATASET.md                  # FracAtlas dataset information
├── medical_embedder.ipynb      # Main embedding generation notebook
├── cloudinary_urls.json        # Generated URL mapping (after running)
└── Medical_Fractures_Embedbed.h5  # Generated embeddings (after running)
```

## Pipeline Details

### 1. Dataset Download (Cell 3)

Downloads FracAtlas from GitHub:
- Repository: https://github.com/MIPT-Oulu/FracAtlas
- Size: ~500MB
- Images: 4,083 X-ray images
- Format: JPEG/PNG

### 2. Cloudinary Upload (Cell 4)

Uploads all images to Cloudinary CDN:
- Folder: `medical/fractures/`
- Rate limit: ~480 uploads/hour (safe rate)
- Time: ~60 minutes for 4,083 images
- Result: Permanent public URLs

**Example URL**:
```
https://res.cloudinary.com/dxxxxxxxx/image/upload/v1234567890/medical/fractures/femur_001.jpg
```

### 3. BiomedCLIP Model (Cell 5)

Loads Microsoft's BiomedCLIP model:
- Model: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- Training: 15M biomedical image-text pairs from PubMed
- Output: 512-dimensional embeddings
- Device: CUDA if available, else CPU

### 4. Embedding Generation (Cell 6)

Generates embeddings for all images:
- Fetches images from Cloudinary URLs
- Encodes with BiomedCLIP
- L2 normalization for cosine similarity
- Time: ~30-60 minutes for 4,083 images

### 5. HDF5 Storage (Cell 7)

Saves to HDF5 format:
- Embeddings: (N, 512) float32 array
- URLs: (N,) string array
- Metadata: model info, dataset name, etc.
- Compression: gzip level 9
- Size: ~50MB for 4,083 images

### 6. Verification (Cell 8)

Verifies the output:
- Loads HDF5 file
- Checks dimensions
- Displays sample URLs
- Tests similarity computation

## Output Files

### Medical_Fractures_Embedbed.h5

HDF5 file structure:

```python
{
    'embeddings': (4083, 512) float32,  # BiomedCLIP embeddings
    'urls': (4083,) bytes,              # Cloudinary URLs
    'attrs': {
        'model': 'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        'embedding_dim': 512,
        'total_images': 4083,
        'dataset': 'FracAtlas',
        'storage': 'Cloudinary',
        'created_date': '2025-12-01T...'
    }
}
```

### cloudinary_urls.json

JSON mapping file:

```json
[
  {
    "local_path": "FracAtlas/images/fracture_001.jpg",
    "url": "https://res.cloudinary.com/.../fractures/fracture_001.jpg",
    "public_id": "medical/fractures/fracture_001",
    "format": "jpg",
    "size": 125432
  },
  ...
]
```

## Usage with Backend

### Start Medical Search Server

```bash
cd ../backend
conda activate hnsw-backend-venv
python server_medical.py
```

### Test the API

```bash
# Health check
curl http://localhost:5002/health

# Text search
curl -X POST http://localhost:5002/search \
  -H "Content-Type: application/json" \
  -d '{"query": "broken leg bone", "k": 10}'

# Image search
curl -X POST http://localhost:5002/search/image \
  -F "image=@xray.jpg" \
  -F "k=10"
```

### Example Queries

Medical terminology queries that work well:

```python
# Fracture types
"transverse fracture of the tibia"
"spiral fracture of the humerus"
"comminuted fracture of the femur"

# Anatomical locations
"distal radius fracture"
"proximal femur fracture"
"midshaft clavicle fracture"

# Clinical descriptions
"displaced fracture with bone fragments"
"non-displaced hairline fracture"
"pathological fracture due to osteoporosis"

# Simple queries
"broken leg bone"
"fractured arm"
"hip fracture"
```

## Troubleshooting

### Issue: Cloudinary Upload Fails

**Symptoms**: Upload errors, timeout, or rate limit exceeded

**Solutions**:
1. Check Cloudinary credentials
2. Verify internet connection
3. Reduce upload rate (increase `time.sleep()` in Cell 4)
4. Check free tier limits (25GB storage)

### Issue: BiomedCLIP Model Download Fails

**Symptoms**: Timeout or connection error when loading model

**Solutions**:
1. Check internet connection
2. Retry - Hugging Face servers may be busy
3. Download manually:
   ```bash
   python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')"
   ```

### Issue: Out of Memory

**Symptoms**: CUDA out of memory or system RAM exhausted

**Solutions**:
1. Close other applications
2. Use CPU instead of GPU (slower but uses less memory)
3. Process in smaller batches
4. Reduce batch size in Cell 6

### Issue: HDF5 File Not Found in Backend

**Symptoms**: server_medical.py can't find HDF5 file

**Solutions**:
```bash
# Copy file to backend directory
cp Medical_Fractures_Embedbed.h5 ../backend/

# Or set environment variable
export H5_FILE_PATH=/path/to/Medical_Fractures_Embedbed.h5
python server_medical.py
```

### Issue: Images Not Loading in Search Results

**Symptoms**: Search works but images don't display

**Solutions**:
1. Check Cloudinary URLs are public
2. Verify image-proxy endpoint works:
   ```bash
   curl "http://localhost:5002/image-proxy?url=<cloudinary_url>"
   ```
3. Check browser console for CORS errors
4. Verify Cloudinary account is active

## Performance Optimization

### Speed Up Upload

```python
# Increase upload rate (if you have paid Cloudinary account)
time.sleep(0.05)  # ~1200 uploads/hour (risky on free tier)
```

### Speed Up Embedding Generation

```python
# Use GPU
device = "cuda"  # Instead of auto-detection

# Batch processing (requires more memory)
batch_size = 16  # Process 16 images at once
```

### Reduce HDF5 File Size

```python
# Use higher compression
f.create_dataset('embeddings', data=embeddings, 
                 compression='gzip', compression_opts=9)

# Or use float16 instead of float32 (less precise)
embeddings = embeddings.astype(np.float16)
```

## Advanced Usage

### Custom Dataset

To use your own X-ray images:

1. Organize images in a folder
2. Modify Cell 3 to point to your folder
3. Ensure images are de-identified (no patient info)
4. Follow HIPAA compliance if applicable

### Different Fracture Types

To organize by fracture type:

```python
# Upload to different folders
cloudinary.uploader.upload(img, folder=f"medical/fractures/{fracture_type}")
```

### Multiple Datasets

To combine multiple datasets:

1. Run notebook for each dataset separately
2. Merge HDF5 files:

```python
import h5py
import numpy as np

# Load all files
embeddings_list = []
urls_list = []

for h5_file in ['dataset1.h5', 'dataset2.h5']:
    with h5py.File(h5_file, 'r') as f:
        embeddings_list.append(f['embeddings'][:])
        urls_list.append(f['urls'][:])

# Merge
all_embeddings = np.concatenate(embeddings_list)
all_urls = np.concatenate(urls_list)

# Save merged file
with h5py.File('merged.h5', 'w') as f:
    f.create_dataset('embeddings', data=all_embeddings)
    f.create_dataset('urls', data=all_urls)
```

## Next Steps

After generating embeddings:

1. ✅ Copy HDF5 file to `backend/` directory
2. ✅ Start `server_medical.py` on port 5002
3. ✅ Test with example queries
4. ✅ Integrate with frontend (optional)
5. ✅ Deploy to production

## Support

- **Cloudinary Issues**: See `CLOUDINARY_SETUP.md`
- **Dataset Issues**: See `DATASET.md`
- **Backend Issues**: See `../backend/README.md`
- **Model Issues**: https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224

## License

- **Code**: MIT License (this project)
- **FracAtlas Dataset**: Check repository license
- **BiomedCLIP Model**: Apache 2.0 License
- **Cloudinary**: Free tier terms of service

---

**Ready to generate embeddings?** Follow the Quick Start guide above!

