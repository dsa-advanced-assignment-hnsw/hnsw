# HNSW Semantic Search Engine

A powerful multi-modal semantic search engine for images, papers, and medical diagnostics using HNSW (Hierarchical Navigable Small World) algorithm.

**Technologies:** Python 3.10+ | PyTorch 2.0+ | Next.js 16 | Flask 3.0 | TypeScript 5.0+

**License:** MIT | **Status:** Maintained

## Features

### Image Search
- **Natural Language Search**: Search images using plain English queries
- **Image-to-Image Search**: Upload an image to find similar ones
- **CLIP Embeddings**: State-of-the-art vision-language AI model (ViT-B/32)
- **Multi-Source Support**: 1000+ images from Flickr, Pinterest, Google, Meta, Reddit
- **Fast Retrieval**: HNSW algorithm for sub-100ms search times
- **LRU Caching**: Intelligent image caching for improved performance

### Paper Search
- **Semantic Paper Search**: Search 100K-1M arXiv papers semantically
- **Document Upload**: Upload PDF/TXT/MD files to find similar papers
- **High-Quality Embeddings**: Sentence Transformers (all-roberta-large-v1, 1024-dim)
- **Comprehensive Coverage**: Full arXiv metadata indexed
- **Fast Retrieval**: Sub-200ms search across millions of papers

### Medical Search
- **Bone Fracture Search**: Search X-ray images by medical terms
- **BiomedCLIP Model**: Specialized medical vision-language AI (512-dim)
- **Local Secure Storage**: Privacy-first local image serving
- **Clinical Accuracy**: Finds similar fracture patterns instantly
- **Efficient Indexing**: ~19MB storage for 3,300+ X-rays

### UI Features
- Modern responsive design with dark/light mode
- Real-time similarity scores
- Smart image caching
- Drag-and-drop file upload
- Mobile-friendly interface

## Architecture

The system consists of three main layers:

### Frontend Layer (Next.js + TypeScript)
- React 19 components with TypeScript
- Custom hooks for search, image upload, and data management
- Responsive UI with Tailwind CSS
- Dark/light theme support
- Client-side image caching

### Backend Layer (Flask)
Three separate Flask servers for different search types:
- **Image Search v1** (`server.py`): Local images, base64 serving
- **Image Search v2** (`server_v2.py`): Online images, LRU caching, image proxy
- **Paper Search** (`server_paper.py`): arXiv papers, document upload support
- **Medical Search** (`server_medical.py`): X-ray images, local secure serving
- **Unified Backend** (`app.py`): Modular architecture with shared core components

### ML & Storage Layer
**Models:**
- CLIP ViT-B/32 (512-dim) for image search
- Sentence Transformers all-roberta-large-v1 (1024-dim) for paper search
- BiomedCLIP (512-dim) for medical image search

**Storage:**
- HDF5 files for pre-computed embeddings
- HNSW binary indexes for fast retrieval
- Local filesystem for medical images
- LRU cache for online images

## Quick Start

### Prerequisites
- Python 3.10 or higher
- Node.js 18 or higher
- Conda (recommended) or venv
- Pre-computed embedding files (HDF5 format)

### Backend Setup

**Option 1: Using Conda (Recommended)**
```bash
cd backend

# Create and activate environment
conda env create -f environment.yml
conda activate hnsw-backend-venv

# Install dependencies
pip install -r requirements-clean.txt

# Run unified backend (supports all search types)
ENGINE_TYPE=image python app.py  # Port 5000 (default)
# or
ENGINE_TYPE=paper python app.py
# or
ENGINE_TYPE=medical python app.py
```

**Option 2: Using venv**
```bash
cd backend

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-clean.txt

# Run server
python app.py
```

**Legacy Servers (Alternative)**
```bash
python server_v2.py      # Image search v2 (online images)
python server_paper.py   # Paper search
python server_medical.py # Medical search
```

### Frontend Setup

```bash
cd client

# Install dependencies
npm install

# Configure API endpoints
cat > .env.local << EOF
NEXT_PUBLIC_API_URL=http://localhost:5000
NEXT_PUBLIC_PAPER_API_URL=http://localhost:5001
NEXT_PUBLIC_MEDICAL_API_URL=http://localhost:5002
EOF

# Start development server
npm run dev
```

### Access the Application

Open your browser and navigate to **http://localhost:3000**

### Quick Start Scripts

For convenience, use the provided startup scripts:

```bash
# Start backend (from project root)
./start-backend.sh

# Start frontend (in another terminal)
./start-frontend.sh
```

## Technology Stack

### Backend
- **Framework**: Flask 3.0 with CORS support
- **ML Libraries**: PyTorch 2.0+, Transformers, Sentence Transformers
- **Models**: 
  - CLIP ViT-B/32 (OpenAI) - 512-dim embeddings
  - BiomedCLIP - 512-dim medical embeddings
  - all-roberta-large-v1 (Sentence Transformers) - 1024-dim embeddings
- **Vector Search**: hnswlib for approximate nearest neighbor search
- **Data Storage**: HDF5 (h5py) for embeddings, NumPy arrays
- **Image Processing**: Pillow, torchvision
- **Document Processing**: PyPDF2 for PDF text extraction
- **HTTP Client**: requests with connection pooling
- **Production Server**: Gunicorn (optional)

### Frontend
- **Framework**: Next.js 16 with App Router
- **Language**: TypeScript 5.0+
- **UI Library**: React 19
- **Styling**: Tailwind CSS 4
- **Components**: Radix UI primitives
- **Animations**: Framer Motion
- **Icons**: Lucide React
- **Theme**: next-themes for dark/light mode
- **Build Tool**: Turbopack

### HNSW Visualization (Simple-HNSW)
- **Backend**: FastAPI with Python
- **Frontend**: React + Vite + Three.js
- **3D Rendering**: @react-three/fiber, @react-three/drei
- **Language**: TypeScript

## API Endpoints

### Common Endpoints (All Servers)

**Health Check**
```bash
GET /health
```
Returns server status, model info, and configuration.

### Image Search API (Port 5000)

**Search by Text**
```bash
POST /search
Content-Type: application/json

{
  "query": "sunset over mountains",
  "k": 20
}
```

**Search by Image**
```bash
POST /search/image
Content-Type: multipart/form-data

FormData:
  - image: [file]
  - k: 20
```

**Image Proxy (v2 only)**
```bash
GET /image-proxy?url=<encoded_url>
```

**Cache Management (v2 only)**
```bash
GET /cache/stats    # Get cache statistics
POST /cache/clear   # Clear cache
```

### Paper Search API (Port 5001)

**Search by Text**
```bash
POST /search
Content-Type: application/json

{
  "query": "transformer neural networks",
  "k": 20
}
```

**Search by Document**
```bash
POST /search/document
Content-Type: multipart/form-data

FormData:
  - document: [PDF/TXT/MD file]
  - k: 20
```

### Medical Search API (Port 5002)

**Search by Text**
```bash
POST /search
Content-Type: application/json

{
  "query": "distal radius fracture",
  "k": 20
}
```

**Search by X-ray Image**
```bash
POST /search/image
Content-Type: multipart/form-data

FormData:
  - image: [file]
  - k: 20
```

**Serve Local Image**
```bash
GET /image?path=/absolute/path/to/image.jpg
```

For detailed API documentation, see [backend/README.md](backend/README.md).

## Project Structure

```
dsa-advanced-assignment-hnsw/
│
├── backend/                             # Flask Backend Services
│   ├── app.py                          # Unified backend (v3, modular)
│   ├── server_v2.py                    # Image search v2 (online images)
│   ├── server_paper.py                 # Paper search
│   ├── server_medical.py               # Medical search
│   │
│   ├── core/                           # Shared core modules
│   │   ├── base_engine.py             # Base search engine class
│   │   ├── cache.py                   # LRU image cache
│   │   ├── config.py                  # Configuration management
│   │   └── image_fetcher.py           # Image fetching utilities
│   │
│   ├── engines/                        # Search engine implementations
│   │   ├── image_engine.py            # Image search engine
│   │   ├── paper_engine.py            # Paper search engine
│   │   └── medical_engine.py          # Medical search engine
│   │
│   ├── routes/                         # API route blueprints
│   │   ├── search.py                  # Search endpoints
│   │   ├── image_proxy.py             # Image proxy endpoints
│   │   └── health.py                  # Health check endpoints
│   │
│   ├── Data Files (HDF5 + Binary)
│   │   ├── Images_Embedbed_0-100000.h5       # Image embeddings
│   │   ├── Papers_Embedbed_0-100000.h5       # Paper embeddings (100K)
│   │   ├── Medical_Fractures_Embedbed.h5     # Medical embeddings
│   │   └── *.bin                             # HNSW index files
│   │
│   └── Configuration
│       ├── requirements-clean.txt      # Production dependencies
│       ├── requirements.txt            # Full dependencies
│       └── environment.yml             # Conda environment
│
├── client/                              # Next.js Frontend
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx               # Main search interface
│   │   │   ├── layout.tsx             # App layout
│   │   │   └── globals.css            # Global styles
│   │   ├── components/                # React components
│   │   ├── hooks/                     # Custom React hooks
│   │   └── types/                     # TypeScript types
│   ├── .env.local                     # API configuration (not in repo)
│   ├── package.json
│   └── README.md
│
├── simple_hnsw/                        # HNSW Visualization App
│   ├── src/
│   │   └── simple_hnsw/
│   │       ├── hnsw.py                # Pure Python HNSW implementation
│   │       └── distance_metrics.py    # Distance functions
│   ├── web_app/
│   │   ├── backend/
│   │   │   └── server.py              # FastAPI server
│   │   └── frontend/                  # React + Three.js visualization
│   ├── tests/                         # Unit tests and benchmarks
│   └── README.md
│
├── embedders/                          # Embedding Generation Pipelines
│   ├── medical_embedder/
│   │   ├── generate_embeddings_local.py
│   │   ├── bone_fractures/            # Local X-ray dataset
│   │   └── README.md
│   ├── image_embedder/
│   └── paper_embedder/
│
├── notebooks/                          # Jupyter notebooks for analysis
│   ├── 01_data_preparation/
│   ├── 02_embedding_generation/
│   ├── 03_index_building/
│   └── 04_analysis/
│
├── latex/                              # Project report (LaTeX)
│   ├── main.tex
│   ├── sections/
│   └── main.pdf
│
├── Documentation
│   ├── README.md                      # This file
│   ├── CLAUDE.md                      # Development guide
│   ├── QUICKSTART.md                  # Quick start guide
│   └── *.md                           # Various documentation files
│
└── Scripts
    ├── start-backend.sh               # Backend startup script
    ├── start-frontend.sh              # Frontend startup script
    └── start-medical-backend.sh       # Medical backend startup
```

## Performance Metrics

| Metric | Image Search | Paper Search | Medical Search |
|--------|--------------|--------------|----------------|
| **Index Size** | 1,5M images | 100K-1M papers | ~3,400 X-rays |
| **Query Time** | < 100ms | < 200ms | < 50ms |
| **Embedding Dim** | 512 | 1024 | 512 |
| **Storage (HDF5)** | ~5MB | ~4GB (1M) | ~19MB |
| **HNSW Index** | ~2MB | ~400MB (1M) | ~15MB |
| **Model Size** | ~350MB (CLIP) | ~1.3GB (RoBERTa) | ~2GB (BiomedCLIP) |
| **RAM Usage** | ~2GB | ~4-5GB (1M) | ~3GB |
| **Accuracy** | High (CLIP) | High (Semantic) | Clinical (Medical) |

**HNSW Parameters:**
- M: 16-200 (connectivity)
- ef_construction: 200-400 (build quality)
- ef: 200 (search quality)
- Space: Cosine similarity

## Simple-HNSW Visualization

This project includes an interactive 3D visualization tool for understanding the HNSW algorithm:

**Location:** `simple_hnsw/` directory

**Features:**
- Pure Python HNSW implementation for educational purposes
- Interactive 3D graph visualization using React + Three.js
- Real-time animation of node insertion and search
- FastAPI backend serving graph state
- Modern UI with dark/light mode

**Quick Start:**
```bash
cd simple_hnsw

# Linux/macOS
./run_web_app.sh

# Windows
run_web_app.bat
```

Visit **http://localhost:5173** to explore the visualization.

For more details, see [simple_hnsw/README.md](simple_hnsw/README.md).

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Quick start guide with automated scripts
- **[CLAUDE.md](CLAUDE.md)** - Development guide and project context
- **[backend/README.md](backend/README.md)** - Backend API documentation
- **[client/README.md](client/README.md)** - Frontend documentation
- **[simple_hnsw/README.md](simple_hnsw/README.md)** - HNSW visualization guide

## Environment Variables

### Backend (app.py)
```bash
ENGINE_TYPE=image|paper|medical  # Select search engine type
PORT=5000                        # Server port
FLASK_DEBUG=0                    # Debug mode (0 or 1)
IMAGE_CACHE_SIZE_MB=100         # Image cache size
IMAGE_FETCH_TIMEOUT=10          # HTTP timeout for images
MAX_HNSW_ELEMENTS=2000000       # HNSW index capacity
```

### Frontend
```bash
NEXT_PUBLIC_API_URL=http://localhost:5000           # Image search API
NEXT_PUBLIC_PAPER_API_URL=http://localhost:5001     # Paper search API
NEXT_PUBLIC_MEDICAL_API_URL=http://localhost:5002   # Medical search API
```

## Deployment

### Backend Options
1. **VPS/Dedicated Server**: Deploy with Gunicorn + Nginx
2. **Cloud Platforms**: Railway, Render, Google Cloud Run, AWS, Azure
3. **Local + ngrok**: Expose local server to internet

### Frontend Options
1. **Vercel**: Recommended for Next.js (automatic deployment)
2. **Netlify**: Alternative static hosting
3. **Self-hosted**: Build and deploy to any static host

For detailed deployment instructions, see [QUICKSTART.md](QUICKSTART.md) and [backend/README.md](backend/README.md).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

Copyright (c) 2025 HNSW Search Engine

## Built With

CLIP • BiomedCLIP • Sentence Transformers • HNSW • Flask • Next.js • React • TypeScript

