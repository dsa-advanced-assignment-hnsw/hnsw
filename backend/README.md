# HNSW Image Search Backend

Flask API backend for the HNSW Image Search Engine using CLIP embeddings.

## Features

- 🔍 Text-to-image search using CLIP embeddings
- �️ Image-to-image search with file upload support
- �🚀 Fast similarity search with HNSW index
- 🌐 RESTful API with CORS support
- 📊 Pre-computed image embeddings stored in HDF5
- 🔒 Secure file handling with validation

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)
- Pre-computed `images_embeds.h5` file

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Server

### Development
```bash
python server.py
```

The server will start on `http://localhost:5000`

### Production

For production deployment, use a production-grade WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 server:app
```

## API Endpoints

### 1. Search Images by Text
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
      "path": "./images/12345.jpg",
      "score": 0.8542
    }
  ],
  "total": 20
}
```

### 2. Search Images by Image
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
      "path": "./images/12345.jpg",
      "score": 0.8542
    }
  ],
  "total": 20
}
```

### 3. Get Image
**GET** `/image/<path:image_path>`

Retrieve an image by its path (base64 encoded).

**Response:**
```json
{
  "image_data": "data:image/jpeg;base64,...",
  "path": "./images/12345.jpg"
}
```

### 3. Health Check
**GET** `/health`

Check server health and status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
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

- `FLASK_ENV`: Set to `production` for production deployment
- `PORT`: Server port (default: 5000)

## File Structure

```
backend/
├── server.py              # Flask application
├── requirements.txt       # Python dependencies
├── images_embeds.h5      # Pre-computed embeddings
└── search_using_hnsw.ipynb # Jupyter notebook for reference
```

## Notes

- The `images_embeds.h5` file must be present for the server to work
- CLIP model will be downloaded on first run (~350MB)
- GPU acceleration is automatic if CUDA is available
- CORS is enabled for all origins (configure in production)

## Troubleshooting

### Server won't start
- Check if port 5000 is already in use
- Verify `images_embeds.h5` exists
- Ensure all dependencies are installed

### Out of memory
- Reduce batch size in search queries
- Use CPU instead of GPU
- Increase system memory

### CORS errors
- Check that CORS is properly configured in `server.py`
- Verify frontend API URL matches backend URL 