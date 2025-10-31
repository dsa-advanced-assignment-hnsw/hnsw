from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import clip
import numpy as np
import hnswlib
import h5py
import os
from PIL import Image
import base64
import io

app = Flask(__name__)
# Enable CORS with explicit configuration for Ngrok compatibility
CORS(app, 
     origins=['*'],  # Allow all origins (restrict in production)
     allow_headers=['Content-Type', 'ngrok-skip-browser-warning'],  # Allow custom headers
    #  allow_headers=['Content-Type'],  # Allow custom headers
     methods=['GET', 'POST', 'OPTIONS'],  # Explicitly allow OPTIONS for preflight
     supports_credentials=False)

class SearchEngine:
    def __init__(self):
        print("🚀 Initializing HNSW Image Search Engine...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📱 Device: {self.device}")
        
        print("📥 Loading CLIP model (this may take a moment)...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        print("✅ CLIP model loaded successfully")
        
        self.index = None
        self.image_paths = None
        self.load_index()
    
    def load_index(self):
        """Load the pre-built HNSW index and image paths"""
        try:
            print("📂 Loading image embeddings from HDF5 file...")
            # Load embeddings and paths
            with h5py.File("images_embeds.h5", "r") as f:
                self.image_paths = f["image_path"][:]
                print(f"   → Loaded {len(self.image_paths)} image paths")
                embs = f["embeddings"][:]
                print(f"   → Loaded embeddings shape: {embs.shape}")
            
            # Initialize HNSW index
            print("🔧 Building HNSW index...")
            dim = embs.shape[1]
            self.index = hnswlib.Index(space='cosine', dim=dim)
            self.index.init_index(max_elements=int(5e5), ef_construction=400, M=200)
            self.index.set_ef(200)
            self.index.add_items(embs)
            
            print(f"✅ Search engine ready! Loaded {len(self.image_paths)} images")
            print(f"📊 Index dimension: {dim}, Device: {self.device}")
        except Exception as e:
            print(f"❌ Error loading index: {e}")
            import traceback
            traceback.print_exc()
    
    def text_to_vector(self, text):
        """Convert text query to vector"""
        text_tokens = clip.tokenize([text.strip()]).to(self.device)
        with torch.no_grad():
            vector_text = self.model.encode_text(text_tokens)
            vector_text = vector_text / vector_text.norm(dim=-1, keepdim=True)
        return vector_text.cpu().numpy().astype(np.float32)
    
    def search(self, query, k=20):
        """Search for similar images using text query"""
        if self.index is None:
            return []
        
        try:
            # Convert query to vector
            query_vector = self.text_to_vector(query)
            
            # Search in HNSW index
            indices, distances = self.index.knn_query(query_vector, k=k)
            
            # Get image paths and scores
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                image_path = self.image_paths[idx].decode('utf-8') if isinstance(self.image_paths[idx], bytes) else self.image_paths[idx]
                similarity_score = 1 - distance  # Convert distance to similarity
                results.append({
                    'path': image_path,
                    'score': float(similarity_score)
                })
            
            return results
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def image_to_vector(self, image):
        """Convert PIL image to vector"""
        try:
            # Preprocess image and convert to tensor
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                vector_image = self.model.encode_image(image_input)
                vector_image = vector_image / vector_image.norm(dim=-1, keepdim=True)
            
            return vector_image.cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"❌ Image encoding error: {e}")
            return None
    
    def search_by_image(self, image, k=20):
        """Search for similar images using uploaded image"""
        if self.index is None:
            return []
        
        try:
            # Convert image to vector
            query_vector = self.image_to_vector(image)
            if query_vector is None:
                return []
            
            # Search in HNSW index
            indices, distances = self.index.knn_query(query_vector, k=k)
            
            # Get image paths and scores
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                image_path = self.image_paths[idx].decode('utf-8') if isinstance(self.image_paths[idx], bytes) else self.image_paths[idx]
                similarity_score = 1 - distance  # Convert distance to similarity
                results.append({
                    'path': image_path,
                    'score': float(similarity_score)
                })
            
            return results
        except Exception as e:
            print(f"❌ Image search error: {e}")
            return []

# Initialize search engine
print("\n" + "="*60)
print("  HNSW IMAGE SEARCH BACKEND")
print("="*60 + "\n")

search_engine = SearchEngine()

print("\n" + "="*60)
print("  Server initialization complete!")
print("="*60 + "\n")

@app.route('/search', methods=['POST'])
def search_images():
    """API endpoint for text-based image search"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        k = min(data.get('k', 20), 100)  # Limit to max 100 results
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        # Perform search
        results = search_engine.search(query, k)
        
        return jsonify({
            'query': query,
            'query_type': 'text',
            'results': results,
            'total': len(results)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search/image', methods=['POST'])
def search_by_image():
    """API endpoint for image-based search"""
    try:
        # Check if image file is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Get k parameter from form data
        k = min(int(request.form.get('k', 20)), 100)
        
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
        if file_ext not in allowed_extensions:
            return jsonify({'error': 'Invalid file type. Supported formats: PNG, JPG, JPEG, GIF, BMP, WEBP'}), 400
        
        # Open and process the image
        try:
            image = Image.open(file.stream).convert('RGB')
        except Exception as e:
            return jsonify({'error': f'Cannot process image: {str(e)}'}), 400
        
        # Perform search
        results = search_engine.search_by_image(image, k)
        
        return jsonify({
            'query': file.filename,
            'query_type': 'image',
            'results': results,
            'total': len(results)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/image/<path:image_path>')
def serve_image(image_path):
    """Serve images from the dataset"""
    try:
        # Security: ensure path is within allowed directories
        if '..' in image_path:
            return jsonify({'error': 'Invalid path'}), 400
        
        # Get the directory where server.py is located (backend/)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Handle relative paths (e.g., "./images/123.jpg" or "images/123.jpg")
        if image_path.startswith('./'):
            image_path = image_path[2:]  # Remove "./" prefix
        
        # Construct absolute path
        full_path = os.path.join(base_dir, image_path)
        
        # Verify path is within the backend directory (security check)
        if not os.path.abspath(full_path).startswith(base_dir):
            return jsonify({'error': 'Invalid path - outside allowed directory'}), 400
            
        if not os.path.exists(full_path):
            return jsonify({
                'error': 'Image not found',
                'path': image_path,
                'full_path': full_path,
                'message': 'The image file does not exist at the specified location'
            }), 404
        
        # Verify it's actually an image file
        if not full_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
            return jsonify({
                'error': 'Invalid file type',
                'path': image_path,
                'message': 'File is not a supported image format'
            }), 400
        
        # Convert image to base64 for web display
        with open(full_path, 'rb') as img_file:
            img_data = img_file.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            
            # Detect image type from extension
            ext = full_path.lower().split('.')[-1]
            mime_type = f"image/{ext}" if ext != 'jpg' else "image/jpeg"
            
        return jsonify({
            'image_data': f"data:{mime_type};base64,{img_base64}",
            'path': image_path
        })
    
    except IOError as e:
        return jsonify({
            'error': 'Cannot read image',
            'path': image_path,
            'message': f'Failed to read image file: {str(e)}'
        }), 500
    
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'path': image_path,
            'message': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': search_engine.index is not None,
        'device': search_engine.device
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # Use environment variable to control debug mode
    # Set FLASK_DEBUG=1 to enable debug mode, otherwise it's disabled
    # debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    debug = False
    
    print(f"🌐 Starting Flask server on http://0.0.0.0:{port}")
    print(f"🔧 Debug mode: {debug}")
    if debug:
        print("⚠️  Debug mode enabled - server will restart on file changes")
    print(f"📡 Access the API at: http://localhost:{port}/health")
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(host='0.0.0.0', port=port, debug=debug)