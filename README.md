# HNSW Image Search Engine

A powerful image search engine that uses natural language queries to find similar images. Built with CLIP embeddings and HNSW (Hierarchical Navigable Small World) algorithm for fast and accurate similarity search.

## 🌟 Features

- 🔍 **Natural Language Search**: Search images using descriptive text queries
- 🖼️ **Image-to-Image Search**: Upload an image to find visually similar images
- ⚡ **Fast Similarity Search**: HNSW algorithm for efficient nearest neighbor search
- 🎨 **Modern UI**: Beautiful, responsive interface built with Next.js and Tailwind CSS
- 🤖 **CLIP Embeddings**: State-of-the-art vision-language model by OpenAI
- 📊 **Similarity Scores**: Visual feedback showing match confidence
- 🌓 **Dark Mode**: Full dark mode support
- 📱 **Dual Search Modes**: Toggle between text and image search seamlessly

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                  Frontend (Next.js)              │
│  • React with TypeScript                         │
│  • Tailwind CSS for styling                      │
│  • Deployed on Vercel                            │
└─────────────────┬───────────────────────────────┘
                  │ REST API
┌─────────────────▼───────────────────────────────┐
│              Backend (Flask)                     │
│  • CLIP model for text & image encoding          │
│  • HNSW index for similarity search              │
│  • HDF5 storage for embeddings                   │
│  • Image upload & processing                     │
│  • Deployed on Railway/Render                    │
└──────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Backend**: Python 3.8+, pip
- **Frontend**: Node.js 18+ or Bun
- **Data**: Pre-computed `images_embeds.h5` file

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/dsa-advanced-assignment-hnsw.git
cd dsa-advanced-assignment-hnsw
```

### 2. Setup Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run server
python server.py
```

Backend will be available at `http://localhost:5000`

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

- **[Backend README](backend/README.md)** - API documentation and backend setup
- **[Frontend README](client/README.md)** - UI customization and frontend development
- **[Deployment Guide](DEPLOYMENT.md)** - Complete deployment instructions for production

## 🎯 How It Works

1. **Image Preprocessing**: Images are converted to embeddings using CLIP ViT-B/32 model
2. **HNSW Index**: Embeddings are indexed using HNSW for efficient similarity search
3. **Text Query**: User's text query is converted to embedding using the same CLIP model
4. **Similarity Search**: HNSW finds k-nearest neighbors based on cosine similarity
5. **Results**: Top matching images are returned with similarity scores

## 🛠️ Technology Stack

### Backend
- **Framework**: Flask 3.0
- **ML Model**: OpenAI CLIP (ViT-B/32)
- **Vector Search**: hnswlib
- **Data Storage**: HDF5 (h5py)
- **Deep Learning**: PyTorch

### Frontend
- **Framework**: Next.js 15
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Deployment**: Vercel

## 📊 API Endpoints

### Search Images by Text
```bash
POST /search
Content-Type: application/json

{
  "query": "beach sunset",
  "k": 20
}
```

### Search Images by Image
```bash
POST /search/image
Content-Type: multipart/form-data

FormData:
- image: [image file]
- k: 20
```

### Get Image
```bash
GET /image/:path
```

### Health Check
```bash
GET /health
```

## 🚢 Deployment

### Quick Deploy

1. **Backend to Railway:**
   ```bash
   # Connect repo to Railway and deploy
   ```

2. **Frontend to Vercel:**
   ```bash
   cd client
   vercel
   ```

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

## 📝 Project Structure

```
dsa-advanced-assignment-hnsw/
├── backend/
│   ├── server.py              # Flask API server
│   ├── requirements.txt       # Python dependencies
│   ├── images_embeds.h5      # Pre-computed embeddings
│   ├── search_using_hnsw.ipynb # Research notebook
│   └── README.md              # Backend documentation
├── client/
│   ├── src/
│   │   └── app/
│   │       └── page.tsx       # Main search interface
│   ├── package.json           # Node dependencies
│   ├── vercel.json           # Vercel configuration
│   └── README.md              # Frontend documentation
├── DEPLOYMENT.md              # Deployment guide
└── README.md                  # This file
```

## 🧪 Example Queries

Try these search queries:
- "dog playing in park"
- "beach sunset"
- "mountain landscape"
- "city skyline at night"
- "cat sleeping"

## 🔧 Configuration

### Backend Configuration

Edit `backend/server.py`:
```python
# Change port
app.run(host='0.0.0.0', port=5000)

# Configure CORS
CORS(app, origins=["https://your-domain.com"])
```

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
- [hnswlib](https://github.com/nmslib/hnswlib) for fast approximate nearest neighbor search
- [Next.js](https://nextjs.org/) for the frontend framework
- [Flask](https://flask.palletsprojects.com/) for the backend framework

## 📞 Support

- 📧 Email: your-email@example.com
- 💬 Issues: [GitHub Issues](https://github.com/yourusername/dsa-advanced-assignment-hnsw/issues)

## ⭐ Star History

If you find this project useful, please consider giving it a star!

---

**Built with ❤️ using CLIP, HNSW, Flask, and Next.js** 