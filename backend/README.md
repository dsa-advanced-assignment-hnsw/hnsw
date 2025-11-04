# HNSW Search Backend

Flask API backend for the HNSW Semantic Search Engine supporting both image search (using CLIP embeddings) and paper search (using Sentence Transformers).

## Features

### Image Search (server.py & server_v2.py)
- 🔍 Text-to-image search using CLIP embeddings
- 🖼️ Image-to-image search with file upload support
- 🌐 Multi-source online image support (v2): Flickr, Pinterest, Google, Meta, Reddit
- 💾 LRU image caching (v2) for improved performance
- 🖥️ Local image serving (v1) or image proxy (v2)

### Paper Search (server_paper.py - NEW)
- 📄 Text-to-paper search using Sentence Transformers
- 📚 Document-to-paper search (upload PDF, TXT, or MD files)
- 🔬 Semantic search across 100K-1M arXiv papers
- 📖 PDF text extraction for document queries

### General
- ⚡ Fast similarity search with HNSW index
- 🌐 RESTful API with CORS support
- 📊 Pre-computed embeddings stored in HDF5
- 🔒 Secure file handling with validation

## Prerequisites

- Python 3.8+
- Conda (recommended for environment management)
- CUDA-compatible GPU (optional, for faster processing)
- Pre-computed embedding files:
  - Image search: `images_embeds.h5` (v1) or `images_embeds_new.h5` (v2)
  - Paper search: `Papers_Embedbed_0-100000.h5` or `Papers_Embedbed_0-1000000.h5`

## Installation

**Recommended:** Use conda for environment management

1. Activate conda environment:
```bash
conda activate hnsw-backend-venv
```

2. Install dependencies:
```bash
pip install -r requirements-clean.txt
```

**Alternative:** Use virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements-clean.txt
```

## Running the Server

### Development

**Image Search v1 (Local Images - ~100 images)**
```bash
python server.py
```

**Image Search v2 (Online Images - ~1000 images, RECOMMENDED)**
```bash
python server_v2.py
```

**Paper Search (NEW - 100K-1M papers)**
```bash
python server_paper.py
```

The server will start on `http://localhost:5000`

**Note:** You can only run one server at a time on the same port. To run multiple servers simultaneously, modify the port in the code or use environment variables.

### Production

For production deployment, use a production-grade WSGI server like Gunicorn:

```bash
pip install gunicorn

# For image search v2
gunicorn -w 1 -b 0.0.0.0:5000 server_v2:app --timeout 120

# For paper search
gunicorn -w 1 -b 0.0.0.0:5000 server_paper:app --timeout 120
```

**Note:** Use `-w 1` (single worker) to avoid memory issues with large models.

## API Endpoints

### Image Search API (server.py & server_v2.py)

#### 1. Search Images by Text
**POST** `/search`

Search for images using text query.

**Request Body:**
```json
{
  "query": "beach sunset",
  "k": 20
}
```

**Response:**
```json
{
  "query": "beach sunset",
  "query_type": "text",
  "results": [
    {
      "path": "./images/12345.jpg",    // v1: local path
      "url": "https://...",             // v2: image URL
      "score": 0.8542
    }
  ],
  "total": 20
}
```

#### 2. Search Images by Image
**POST** `/search/image`

Search for similar images using an uploaded image.

**Request (multipart/form-data):**
- `image`: Image file (PNG, JPG, JPEG, GIF, BMP, WEBP)
- `k`: Number of results (optional, default: 20, max: 100)

**Response:**
```json
{
  "query": "uploaded_image.jpg",
  "query_type": "image",
  "results": [
    {
      "path": "./images/12345.jpg",    // v1
      "url": "https://...",             // v2
      "score": 0.8542
    }
  ],
  "total": 20
}
```

#### 3. Get Image (v1 only)
**GET** `/image/<path:image_path>`

Retrieve an image by its path (base64 encoded).

**Response:**
```json
{
  "image_data": "data:image/jpeg;base64,...",
  "path": "./images/12345.jpg"
}
```

#### 4. Image Proxy (v2 only)
**GET** `/image-proxy?url=<encoded_url>`

Fetch and proxy an image from online source.

**Query Parameters:**
- `url`: URL-encoded image URL

**Response:** Image binary data

#### 5. Cache Management (v2 only)
**GET** `/cache/stats`

Get image cache statistics.

**Response:**
```json
{
  "size": 52428800,
  "size_mb": 50.0,
  "limit_mb": 100,
  "usage_percent": 50.0,
  "count": 42
}
```

**POST** `/cache/clear`

Clear the image cache.

### Paper Search API (server_paper.py)

#### 1. Search Papers by Text
**POST** `/search`

Search for papers using text query.

**Request Body:**
```json
{
  "query": "deep learning transformers",
  "k": 20
}
```

**Response:**
```json
{
  "query": "deep learning transformers",
  "query_type": "text",
  "results": [
    {
      "url": "https://arxiv.org/pdf/1234.5678",
      "score": 0.8542
    }
  ],
  "total": 20
}
```

#### 2. Search Papers by Document
**POST** `/search/document`

Search for similar papers using an uploaded document.

**Request (multipart/form-data):**
- `document`: Document file (TXT, PDF, MD)
- `k`: Number of results (optional, default: 20, max: 100)

**Response:**
```json
{
  "query": "uploaded_paper.pdf",
  "query_type": "document",
  "results": [
    {
      "url": "https://arxiv.org/pdf/1234.5678",
      "score": 0.8542
    }
  ],
  "total": 20
}
```

### Common Endpoints

#### Health Check
**GET** `/health`

Check server health and status.

**Response (Image Search):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "server_type": "image_search",
  "version": "v2",
  "cache_stats": {...}              // v2 only
}
```

**Response (Paper Search):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "server_type": "paper_search",
  "total_papers": 1000000
}
```

## Deployment Options

### Option 1: Cloud Services (Recommended)

#### Railway
1. Create account at [Railway.app](https://railway.app)
2. Create new project
3. Connect GitHub repository
4. Add environment variables
5. Deploy

#### Render
1. Create account at [Render.com](https://render.com)
2. Create new Web Service
3. Connect GitHub repository
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `gunicorn -w 4 -b 0.0.0.0:$PORT server:app`
6. Deploy

#### Google Cloud Run / AWS / Azure
Follow platform-specific Docker deployment guides.

### Option 2: VPS/Dedicated Server

1. Set up server with Python 3.8+
2. Clone repository
3. Install dependencies
4. Set up systemd service or use PM2
5. Configure nginx reverse proxy
6. Enable HTTPS with Let's Encrypt

### Option 3: Local Network

Keep the server running on your local machine and expose it:
```bash
# Using ngrok
ngrok http 5000

# Update frontend NEXT_PUBLIC_API_URL with ngrok URL
```

## Environment Variables

### Image Search v2 (server_v2.py)
- `PORT`: Server port (default: 5000)
- `FLASK_DEBUG`: Enable debug mode (default: 0)
- `H5_FILE_PATH`: Path to HDF5 file (default: images_embeds_new.h5)
- `MAX_HNSW_ELEMENTS`: HNSW index capacity (default: 2000000)
- `IMAGE_CACHE_SIZE_MB`: Image cache size in MB (default: 100)
- `IMAGE_FETCH_TIMEOUT`: HTTP timeout in seconds (default: 10)

### Image Search v1 (server.py)
- `PORT`: Server port (default: 5000)
- `FLASK_DEBUG`: Enable debug mode (default: 0)

### Paper Search (server_paper.py)
- `PORT`: Server port (default: 5000)
- `FLASK_DEBUG`: Enable debug mode (default: 0)
- HDF5 file path must be changed in code to switch between 100K/1M datasets

## File Structure

```
backend/
├── server.py                      # Flask API - Image search v1 (local)
├── server_v2.py                   # Flask API - Image search v2 (online)
├── server_paper.py                # Flask API - Paper search (NEW)
├── requirements-clean.txt         # Python dependencies (recommended)
├── requirements.txt               # Legacy dependencies
├── images_embeds.h5              # Image embeddings v1 (~100 images)
├── images_embeds_new.h5          # Image embeddings v2 (~1000 images)
├── Papers_Embedbed_0-10000.h5    # Paper embeddings (10K papers, for testing)
├── Papers_Embedbed_0-100000.h5   # Paper embeddings (100K papers)
├── Papers_Embedbed_0-1000000.h5  # Paper embeddings (1M papers, 4.1GB)
├── search_using_hnsw.ipynb       # Research notebook
└── test_image_paths.py           # Test utility for HDF5 files
```

## Notes

### Image Search
- HDF5 embedding files must be present for servers to work
- v1: `images_embeds.h5` (~235KB, ~100 local images)
- v2: `images_embeds_new.h5` (~4MB, ~1000 online images)
- CLIP model will be downloaded on first run (~350MB)
- GPU acceleration is automatic if CUDA is available
- v2 includes LRU cache for better performance with online images

### Paper Search
- HDF5 file required: `Papers_Embedbed_0-100000.h5` or `Papers_Embedbed_0-1000000.h5`
- Sentence Transformer model will be downloaded on first run (~1.3GB)
- 1M papers dataset requires ~4.1GB HDF5 file + ~3-4GB RAM for HNSW index
- Supports PDF, TXT, and MD file uploads for document-based search

### General
- CORS is enabled for all origins (configure in production)
- Always use conda environment: `conda activate hnsw-backend-venv`
- Only one server can run on port 5000 at a time

## Troubleshooting

### Server won't start
- Check if port 5000 is already in use: `lsof -i :5000` or `netstat -an | grep 5000`
- Verify HDF5 embedding files exist in the backend directory
- Ensure conda environment is activated: `conda activate hnsw-backend-venv`
- Ensure all dependencies are installed: `pip install -r requirements-clean.txt`
- Check Python version: Python 3.8+ required

### Out of memory
- **Image Search**: Reduce number of indexed images or use smaller HDF5 file
- **Paper Search**: Use 100K dataset instead of 1M, or increase system RAM (3-4GB minimum for 1M papers)
- Reduce batch size in search queries
- Use CPU instead of GPU (edit device selection in code)
- Close other applications to free memory

### Model download issues
- **CLIP**: ~350MB model will download from OpenAI on first run
- **Sentence Transformers**: ~1.3GB model will download from Hugging Face
- Check internet connection and firewall settings
- Set HuggingFace cache: `export TRANSFORMERS_CACHE=/path/to/cache`

### CORS errors
- Check that CORS is properly configured in server files (already set to `origins=['*']`)
- Verify frontend API URL matches backend URL
- Check browser console for detailed error messages

### Image fetch failures (v2 only)
- Check network connectivity
- Verify image URLs are valid and accessible
- Some platforms may block automated requests
- Increase `IMAGE_FETCH_TIMEOUT` if images are slow to load
- Clear cache if seeing stale data: `POST /cache/clear`

### PDF extraction errors (paper search)
- Ensure PyPDF2 is installed: `pip install PyPDF2`
- Some PDFs may be encrypted or have extraction restrictions
- Try converting PDF to TXT before uploading
- Check uploaded file is valid PDF format 