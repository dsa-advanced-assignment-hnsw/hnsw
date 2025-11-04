# HNSW Semantic Search Engine

A powerful semantic search engine that enables searching both images and scientific papers using natural language queries. Built with state-of-the-art ML models (CLIP for images, Sentence Transformers for papers) and HNSW (Hierarchical Navigable Small World) algorithm for fast and accurate similarity search.

## 🌟 Features

### Image Search
- 🔍 **Natural Language Search**: Search images using descriptive text queries
- 🖼️ **Image-to-Image Search**: Upload an image to find visually similar images
- 🤖 **CLIP Embeddings**: State-of-the-art vision-language model by OpenAI
- 🌐 **Multi-Source Support**: Search across 1000+ images from Flickr, Pinterest, Google, Meta, Reddit, and more
- 🖥️ **Dual Versions**: v1 for local images (~100), v2 for online images (~1000, scalable to 1.5M)

### Paper Search (NEW)
- 📄 **Scientific Paper Search**: Search through 1M+ arXiv papers using semantic queries
- 📚 **Document Upload**: Upload text, PDF, or Markdown files to find similar research papers
- 🔬 **Sentence Transformers**: High-quality embeddings using all-roberta-large-v1 model
- 🎓 **Comprehensive Coverage**: Full arXiv metadata with ~1 million papers indexed

### General
- ⚡ **Fast Similarity Search**: HNSW algorithm for efficient nearest neighbor search
- 🎨 **Modern UI**: Beautiful, responsive interface built with Next.js and Tailwind CSS
- 📊 **Similarity Scores**: Visual feedback showing match confidence
- 🌓 **Dark Mode**: Full dark mode support
- 📱 **Flexible Search Modes**: Toggle between different search types seamlessly

## 🏗️ Architecture

```
┌───────────────────────────────────────────────────────────┐
│                    Frontend (Next.js)                      │
│  • React with TypeScript                                   │
│  • Tailwind CSS for styling                                │
│  • Dual search interfaces (images & papers)                │
│  • Deployed on Vercel                                      │
└────────────────────┬──────────────────────────────────────┘
                     │ REST API
┌────────────────────┴──────────────────────────────────────┐
│                  Backend Services (Flask)                  │
├────────────────────────────┬───────────────────────────────┤
│   Image Search (v1 & v2)   │     Paper Search (NEW)        │
├────────────────────────────┼───────────────────────────────┤
│ • CLIP (ViT-B/32)          │ • Sentence Transformers       │
│ • Local/Online images      │   (all-roberta-large-v1)      │
│ • HNSW index (~1000 imgs)  │ • HNSW index (~1M papers)     │
│ • Image proxy & caching    │ • PDF/Text extraction         │
│ • HDF5 storage             │ • HDF5 storage (4.1GB)        │
└────────────────────────────┴───────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Backend**: Python 3.8+, conda (recommended)
- **Frontend**: Node.js 18+ or Bun
- **Data**:
  - Image Search: `images_embeds.h5` (v1, ~100 images) or `images_embeds_new.h5` (v2, ~1000 images)
  - Paper Search: `Papers_Embedbed_0-100000.h5` or `Papers_Embedbed_0-1000000.h5` (1M papers, 4.1GB)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/dsa-advanced-assignment-hnsw.git
cd dsa-advanced-assignment-hnsw
```

### 2. Setup Backend

**Note:** This project uses conda for environment management.

```bash
cd backend

# Activate conda environment
conda activate hnsw-backend-venv

# Install dependencies (if needed)
pip install -r requirements-clean.txt

# Run image search server (v1 - local images)
python server.py

# OR run image search server (v2 - online images, RECOMMENDED)
python server_v2.py

# OR run paper search server (NEW)
python server_paper.py
```

Backend will be available at `http://localhost:5000`

**Choose the appropriate server:**
- `server.py` - For local image search (~100 images)
- `server_v2.py` - For online image search (~1000 images, multi-source)
- `server_paper.py` - For scientific paper search (~100K-1M papers)

### 3. Setup Frontend

```bash
cd client

# Install dependencies
yarn install  # or npm install

# Create environment file
echo "NEXT_PUBLIC_API_URL=http://localhost:5000" > .env.local

# Run development server
yarn dev  # or npm run dev
```

Frontend will be available at `http://localhost:3000`

### 4. Open in Browser

Visit [http://localhost:3000](http://localhost:3000) and start searching!

## 📚 Documentation

- **[Backend README](backend/README.md)** - API documentation and backend setup for all servers
- **[Frontend README](client/README.md)** - UI customization and frontend development
- **[Paper Embedder README](paper_embedder/README.md)** - Generate embeddings for arXiv papers
- **[Image Embedder README](image_embedder/README.md)** - Generate embeddings for Open Images V7 dataset
- **[CLAUDE.md](CLAUDE.md)** - Comprehensive development guide and architecture details

## 🎯 How It Works

### Image Search
1. **Image Preprocessing**: Images are converted to embeddings using CLIP ViT-B/32 model
2. **HNSW Index**: Embeddings are indexed using HNSW for efficient similarity search
3. **Text/Image Query**: User's query (text or image) is converted to embedding using CLIP
4. **Similarity Search**: HNSW finds k-nearest neighbors based on cosine similarity
5. **Results**: Top matching images are returned with similarity scores

### Paper Search
1. **Paper Preprocessing**: Paper abstracts are converted to embeddings using Sentence Transformers
2. **HNSW Index**: 1M+ paper embeddings are indexed for fast retrieval
3. **Text/Document Query**: User's query or uploaded document is converted to embedding
4. **Similarity Search**: HNSW finds k-nearest papers based on semantic similarity
5. **Results**: Top matching papers with arXiv URLs and similarity scores

## 🛠️ Technology Stack

### Backend
- **Framework**: Flask 3.0
- **ML Models**:
  - OpenAI CLIP (ViT-B/32) for images - 512-dim embeddings
  - Sentence Transformers (all-roberta-large-v1) for papers - 1024-dim embeddings
- **Vector Search**: hnswlib (HNSW algorithm)
- **Data Storage**: HDF5 (h5py)
- **Deep Learning**: PyTorch
- **Document Processing**: PyPDF2 for PDF text extraction
- **Environment**: Conda for dependency management

### Frontend
- **Framework**: Next.js 15 (with Turbopack)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Deployment**: Vercel

### Data Pipeline
- **Image Embedder**: Jupyter Notebook for Open Images V7 dataset processing
- **Paper Embedder**: Jupyter Notebook for arXiv metadata processing

## 📊 API Endpoints

### Image Search API (server.py / server_v2.py)

#### Search Images by Text
```bash
POST /search
Content-Type: application/json

{
  "query": "beach sunset",
  "k": 20
}
```

#### Search Images by Image
```bash
POST /search/image
Content-Type: multipart/form-data

FormData:
- image: [image file]
- k: 20
```

#### Get Image (v1)
```bash
GET /image/<path>
```

#### Image Proxy (v2)
```bash
GET /image-proxy?url=<encoded_url>
```

#### Cache Management (v2)
```bash
GET /cache/stats      # Get cache statistics
POST /cache/clear     # Clear image cache
```

### Paper Search API (server_paper.py)

#### Search Papers by Text
```bash
POST /search
Content-Type: application/json

{
  "query": "deep learning transformers",
  "k": 20
}
```

#### Search Papers by Document
```bash
POST /search/document
Content-Type: multipart/form-data

FormData:
- document: [.txt, .pdf, or .md file]
- k: 20
```

### Common Endpoints

#### Health Check
```bash
GET /health
```

## 🚢 Deployment

### Frontend (Vercel - Recommended)
```bash
cd client
vercel
# Set environment variable: NEXT_PUBLIC_API_URL=https://your-backend-url.com
```

### Backend Options

**Option 1: Local/VPS with Conda**
```bash
conda activate hnsw-backend-venv
python server_v2.py    # or server_paper.py
# For production: gunicorn -w 1 -b 0.0.0.0:5000 server_v2:app --timeout 120
```

**Option 2: Cloud Platforms (Railway, Render, etc.)**
- Deploy backend directory
- Set environment variables (see CLAUDE.md)
- Use gunicorn for production
- Ensure sufficient memory (3-4GB for large datasets)

**Important:** CORS is pre-configured for cross-origin requests, allowing separate frontend/backend deployment.

## 📝 Project Structure

```
dsa-advanced-assignment-hnsw/
├── backend/
│   ├── server.py                      # Flask API - Image search v1 (local)
│   ├── server_v2.py                   # Flask API - Image search v2 (online)
│   ├── server_paper.py                # Flask API - Paper search (NEW)
│   ├── requirements-clean.txt         # Python dependencies (recommended)
│   ├── images_embeds.h5              # Image embeddings v1 (~100 images)
│   ├── images_embeds_new.h5          # Image embeddings v2 (~1000 images)
│   ├── Papers_Embedbed_0-100000.h5   # Paper embeddings (100K papers)
│   ├── Papers_Embedbed_0-1000000.h5  # Paper embeddings (1M papers, 4.1GB)
│   ├── search_using_hnsw.ipynb       # Research notebook
│   ├── test_image_paths.py           # Test utility for HDF5 files
│   └── README.md                      # Backend documentation
├── client/
│   ├── src/
│   │   └── app/
│   │       └── page.tsx               # Main search interface
│   ├── package.json                   # Node dependencies
│   ├── vercel.json                   # Vercel configuration
│   └── README.md                      # Frontend documentation
├── paper_embedder/
│   ├── paper_embedder.ipynb          # Generate paper embeddings
│   └── README.md                      # Paper embedder documentation
├── image_embedder/
│   ├── image_embedder.ipynb          # Generate image embeddings
│   └── README.md                      # Image embedder documentation
├── CLAUDE.md                          # Comprehensive development guide
└── README.md                          # This file
```

## 🧪 Example Queries

### Image Search
Try these search queries:
- "dog playing in park"
- "beach sunset"
- "mountain landscape"
- "city skyline at night"
- "cat sleeping"

### Paper Search
Try these research queries:
- "deep learning for computer vision"
- "transformer architecture in natural language processing"
- "reinforcement learning algorithms"
- "quantum computing applications"
- "generative adversarial networks"

## 🔧 Configuration

### Backend Configuration (Environment Variables)

**Image Search v2** (`server_v2.py`):
- `PORT` - Server port (default: 5000)
- `FLASK_DEBUG` - Enable debug mode (default: 0)
- `H5_FILE_PATH` - Path to HDF5 file (default: images_embeds_new.h5)
- `MAX_HNSW_ELEMENTS` - HNSW capacity (default: 2000000)
- `IMAGE_CACHE_SIZE_MB` - Cache size in MB (default: 100)
- `IMAGE_FETCH_TIMEOUT` - HTTP timeout in seconds (default: 10)

**Paper Search** (`server_paper.py`):
- `PORT` - Server port (default: 5000)
- `FLASK_DEBUG` - Enable debug mode (default: 0)
- Update HDF5 file path in code to switch between 100K/1M paper datasets

### Frontend Configuration

Edit `client/.env.local`:
```env
NEXT_PUBLIC_API_URL=https://your-backend-url.com
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [OpenAI CLIP](https://github.com/openai/CLIP) for the vision-language model
- [Sentence Transformers](https://www.sbert.net/) for high-quality text embeddings
- [hnswlib](https://github.com/nmslib/hnswlib) for fast approximate nearest neighbor search
- [arXiv](https://arxiv.org/) for providing open access to scientific papers
- [Open Images V7](https://storage.googleapis.com/openimages/web/index.html) for the image dataset
- [Next.js](https://nextjs.org/) for the frontend framework
- [Flask](https://flask.palletsprojects.com/) for the backend framework

## 📞 Support

- 💬 Issues: [GitHub Issues](https://github.com/dsa-advanced-assignment-hnsw/hnsw-search-engine/issues)
- 📖 Documentation: See [CLAUDE.md](CLAUDE.md) for detailed development guide

## ⭐ Star History

If you find this project useful, please consider giving it a star!

---

**Built with ❤️ using CLIP, Sentence Transformers, HNSW, Flask, and Next.js** 