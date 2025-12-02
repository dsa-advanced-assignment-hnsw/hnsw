<div align="center">

# 🔍 HNSW Semantic Search Engine

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=6366F1&center=true&vCenter=true&width=600&lines=Search+Images+with+Natural+Language;Find+Research+Papers+Instantly;Diagnose+Bone+Fractures+with+AI;CLIP+%2B+OpenCLIP+%2B+HNSW" alt="Typing SVG" />

<p align="center">
  <strong>A powerful multi-modal semantic search engine for images, papers, and medical diagnostics</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Next.js-15-000000?style=for-the-badge&logo=next.js&logoColor=white" alt="Next.js" />
  <img src="https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask" />
  <img src="https://img.shields.io/badge/TypeScript-5.0+-3178C6?style=for-the-badge&logo=typescript&logoColor=white" alt="TypeScript" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=flat-square" alt="License" />
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square" alt="PRs Welcome" />
  <img src="https://img.shields.io/badge/Maintained-Yes-green.svg?style=flat-square" alt="Maintained" />
</p>

---

### 🎯 **[Documentation](./) | [API Docs](./backend/README.md)**

</div>

---

## ✨ Features

<table>
<tr>
<td width="33%">

### 🖼️ Image Search
- 🔍 **Natural Language Search**
  Search images using plain English
- 🖼️ **Image-to-Image Search**
  Upload an image, find similar ones
- 🤖 **CLIP Embeddings**
  State-of-the-art vision-language AI
- 🌐 **Multi-Source Support**
  1000+ images from Flickr, Pinterest, Google
- ⚡ **Lightning Fast**
  HNSW algorithm for instant results

</td>
<td width="33%">

### 📚 Paper Search
- 📄 **Semantic Paper Search**
  Search 1M+ arXiv papers semantically
- 📝 **Document Upload**
  Upload PDF/TXT to find similar papers
- 🔬 **High-Quality Embeddings**
  Sentence Transformers (RoBERTa-large)
- 🎓 **Comprehensive Coverage**
  Full arXiv metadata indexed
- 🚀 **Fast Retrieval**
  Sub-second search across millions

</td>
<td width="33%">

### 🏥 Medical Search (NEW)
- 🦴 **Bone Fracture Search**
  Search X-ray images by medical terms
- 🧠 **OpenCLIP Model**
  ViT-B-16 pretrained on massive datasets
- 🏥 **Local Secure Storage**
  Privacy-first local image serving
- 📊 **Clinical Accuracy**
  Finds similar fracture patterns instantly
- ⚡ **Efficient Indexing**
  ~19MB storage for 3,300+ X-rays

</td>
</tr>
</table>

<div align="center">

### 🎨 **Modern UI** • 🌓 **Dark Mode** • 📱 **Responsive Design** • 📊 **Similarity Scores** • 💾 **Smart Caching**

</div>

---

## 🏗️ Architecture

```mermaid
graph TB
    subgraph Frontend["🎨 Frontend (Next.js + TypeScript)"]
        UI[React Components]
        State[State Management]
        API[API Client]
    end

    subgraph Backend["⚙️ Backend Services (Flask)"]
        ImgV1[Image Search v1<br/>Local Images<br/>Port 5000]
        ImgV2[Image Search v2<br/>Online Images<br/>Port 5000]
        Paper[Paper Search<br/>arXiv Papers<br/>Port 5001]
        Medical[Medical Search<br/>Bone Fractures<br/>Port 5002]
    end

    subgraph ML["🤖 ML Models"]
        CLIP[CLIP ViT-B/32<br/>512-dim]
        ST[Sentence Transformers<br/>1024-dim]
        OpenCLIP[OpenCLIP ViT-B/16<br/>512-dim]
    end

    subgraph Storage["💾 Data Storage"]
        HDF5_Img[images_embeds.h5<br/>~1000 images]
        HDF5_Paper[Papers_Embedbed.h5<br/>~1M papers]
        HDF5_Medical[Medical_Fractures.h5<br/>~3.3K X-rays]
        HNSW_Img[HNSW Index .bin<br/>Fast Retrieval]
        HNSW_Paper[HNSW Index .bin<br/>Fast Retrieval]
        HNSW_Medical[HNSW Index .bin<br/>Fast Retrieval]
        LocalFS[Local Filesystem<br/>Image Storage]
    end

    UI --> API
    API --> ImgV1
    API --> ImgV2
    API --> Paper
    API --> Medical

    ImgV1 --> CLIP
    ImgV2 --> CLIP
    Paper --> ST
    Medical --> OpenCLIP

    CLIP --> HDF5_Img
    ST --> HDF5_Paper
    OpenCLIP --> HDF5_Medical

    HDF5_Img --> HNSW_Img
    HDF5_Paper --> HNSW_Paper
    HDF5_Medical --> HNSW_Medical
    
    Medical --> LocalFS

    style Frontend fill:#6366f1,stroke:#4f46e5,stroke-width:2px,color:#fff
    style Backend fill:#8b5cf6,stroke:#7c3aed,stroke-width:2px,color:#fff
    style ML fill:#ec4899,stroke:#db2777,stroke-width:2px,color:#fff
    style Storage fill:#10b981,stroke:#059669,stroke-width:2px,color:#fff
```

---

## 🚀 Quick Start

<details open>
<summary><b>📦 Installation</b></summary>

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/dsa-advanced-assignment-hnsw/hnsw-search-engine.git
cd hnsw-search-engine
```

### 2️⃣ Setup Backend (Conda Recommended)

```bash
cd backend

# Create environment
conda env create -f environment.yml

# Activate
conda activate hnsw-backend-venv

# Install additional dependencies (if needed)
pip install -r requirements-clean.txt
```

### 3️⃣ Choose Your Server

```bash
# 🖼️ Image Search v2 (Online Images)
python server_v2.py  # Port 5000

# 📄 Paper Search (arXiv Papers)
python server_paper.py  # Port 5001

# 🏥 Medical Search (Bone Fractures)
python server_medical.py  # Port 5002
```

### 4️⃣ Setup Frontend

```bash
cd client

# Install dependencies
npm install

# Configure API URLs
echo "NEXT_PUBLIC_API_URL=http://localhost:5000" > .env.local
echo "NEXT_PUBLIC_PAPER_API_URL=http://localhost:5001" >> .env.local
echo "NEXT_PUBLIC_MEDICAL_API_URL=http://localhost:5002" >> .env.local

# Start dev server
npm run dev
```

### 5️⃣ Open Browser

🎉 Visit **http://localhost:3000** and start searching!

</details>

---

## 🛠️ Technology Stack

<div align="center">

### Backend Technologies

<p>
  <img src="https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenAI-CLIP-412991?style=for-the-badge&logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCLIP-ViT--B16-000000?style=for-the-badge&logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/hnswlib-Vector_Search-FF6B6B?style=for-the-badge" />
  <img src="https://img.shields.io/badge/HDF5-Data_Storage-013243?style=for-the-badge" />
</p>

### Frontend Technologies

<p>
  <img src="https://img.shields.io/badge/Next.js-15-000000?style=for-the-badge&logo=next.js&logoColor=white" />
  <img src="https://img.shields.io/badge/TypeScript-5.0+-3178C6?style=for-the-badge&logo=typescript&logoColor=white" />
  <img src="https://img.shields.io/badge/Tailwind_CSS-3.4-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white" />
  <img src="https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black" />
</p>

### ML Models

<table align="center">
<tr>
<td align="center" width="33%">
  <img src="https://img.icons8.com/color/96/000000/artificial-intelligence.png" width="64" /><br />
  <b>CLIP / OpenCLIP</b><br />
  <sub>Vision-Language Models</sub><br />
  <code>512-dim embeddings</code>
</td>
<td align="center" width="33%">
  <img src="https://img.icons8.com/color/96/000000/machine-learning.png" width="64" /><br />
  <b>Sentence Transformers</b><br />
  <sub>all-roberta-large-v1</sub><br />
  <code>1024-dim embeddings</code>
</td>
<td align="center" width="33%">
  <img src="https://img.icons8.com/color/96/000000/graph.png" width="64" /><br />
  <b>HNSW Algorithm</b><br />
  <sub>Vector Similarity Search</sub><br />
  <code>Sub-second retrieval</code>
</td>
</tr>
</table>

</div>

---

## 📊 API Endpoints

<details>
<summary><b>🏥 Medical Search API (Port 5002)</b></summary>

### Search by Text
```bash
POST /search
Content-Type: application/json

{
  "query": "distal radius fracture",
  "k": 20
}
```

### Search by X-ray Image
```bash
POST /search/image
Content-Type: multipart/form-data

FormData:
  - image: [file]
  - k: 20
```

### Serve Image
```bash
GET /image?path=/absolute/path/to/image.jpg
```
</details>

<details>
<summary><b>🖼️ Image Search API (Port 5000)</b></summary>

### Search by Text
```bash
POST /search
Content-Type: application/json

{
  "query": "sunset over mountains",
  "k": 20
}
```

### Search by Image
```bash
POST /search/image
Content-Type: multipart/form-data

FormData:
  - image: [file]
  - k: 20
```
</details>

<details>
<summary><b>📄 Paper Search API (Port 5001)</b></summary>

### Search by Text
```bash
POST /search
Content-Type: application/json

{
  "query": "transformer neural networks",
  "k": 20
}
```
</details>

---

## 📁 Project Structure

```
dsa-advanced-assignment-hnsw/
│
├── 🎨 client/                           # Next.js Frontend
│   ├── src/app/
│   │   ├── page.tsx                     # Main search interface (All tabs)
│   │   └── layout.tsx                   # App layout
│   ├── .env.local                       # API URL Configuration
│   └── README.md
│
├── ⚙️ backend/                          # Flask Backend Services
│   ├── server.py                        # Image search v1 (local)
│   ├── server_v2.py                     # Image search v2 (online)
│   ├── server_paper.py                  # Paper search (Port 5001)
│   ├── server_medical.py                # Medical search (Port 5002) ⭐
│   │
│   ├── 📦 Data Files
│   │   ├── images_embeds_new.h5         # Image embeddings
│   │   ├── Papers_Embedbed_*.h5         # Paper embeddings
│   │   └── Medical_Fractures_Embedbed.h5 # Medical embeddings
│   │
│   └── 📄 Configuration
│       ├── requirements-clean.txt       # Backend dependencies
│       └── environment.yml              # Conda environment
│
├── 🏥 medical_embedder/                 # Medical Embedding Pipeline
│   ├── generate_embeddings_local.py     # Embedding generator
│   ├── bone_fractures/                  # Local image dataset
│   └── README.md
│
├── 🔬 paper_embedder/                   # Paper Embedding Pipeline
├── 🖼️ image_embedder/                   # Image Embedding Pipeline
│
└── 📚 Documentation
    ├── README.md                        # This file
    └── CLAUDE.md                        # Development guide
```

---

## 📈 Performance

<div align="center">

| Metric | Image Search | Paper Search | Medical Search |
|--------|--------------|--------------|----------------|
| **Index Size** | 1,000 images | 1M papers | ~3,400 X-rays |
| **Query Time** | < 100ms | < 200ms | < 50ms |
| **Embedding** | 512-dim | 1024-dim | 512-dim |
| **Storage** | ~5MB | ~4GB | ~19MB |
| **Accuracy** | High | High | Clinical |

</div>

---

## 📄 License

<div align="center">

**MIT License**

Copyright (c) 2025 HNSW Search Engine

</div>

---

<div align="center">

### 💖 Built with love using

**CLIP • OpenCLIP • HNSW • Flask • Next.js**

</div>
