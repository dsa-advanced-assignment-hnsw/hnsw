from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import open_clip
import numpy as np
import hnswlib
import h5py
import os
from PIL import Image
import base64
import io
import requests
from collections import OrderedDict
import hashlib

app = Flask(__name__)

# Enable CORS for flexible deployment (Vercel frontend + VPS/local backend)
CORS(app,
     origins=['*'],  # Allow all origins - frontend can be on any domain
     allow_headers=['Content-Type', 'ngrok-skip-browser-warning', 'Authorization'],
     methods=['GET', 'POST', 'OPTIONS'],
     supports_credentials=False)

# Configuration from environment variables
IMAGE_CACHE_SIZE_MB = int(os.environ.get('IMAGE_CACHE_SIZE_MB', 100))
IMAGE_FETCH_TIMEOUT = int(os.environ.get('IMAGE_FETCH_TIMEOUT', 10))
H5_FILE_PATH = os.environ.get('H5_FILE_PATH', 'Medical_Fractures_Embedbed.h5')
MAX_HNSW_ELEMENTS = int(os.environ.get('MAX_HNSW_ELEMENTS', 100000))  # 100K capacity for medical images
PREFETCH_IMAGES = os.environ.get('PREFETCH_IMAGES', 'false').lower() == 'true'  # Prefetch images in search results


class LRUImageCache:
    """LRU cache for storing fetched images with size limit"""
    def __init__(self, max_size_bytes):
        self.max_size_bytes = max_size_bytes
        self.current_size = 0
        self.cache = OrderedDict()

    def get(self, key):
        """Get item from cache, move to end (most recently used)"""
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        """Add item to cache, evict oldest if over size limit"""
        value_size = len(value)

        # If single item is larger than cache, don't cache it
        if value_size > self.max_size_bytes:
            return

        # Evict oldest items until we have space
        while self.current_size + value_size > self.max_size_bytes and self.cache:
            _, oldest_value = self.cache.popitem(last=False)
            self.current_size -= len(oldest_value)

        # Add new item
        if key in self.cache:
            self.current_size -= len(self.cache[key])
        self.cache[key] = value
        self.cache.move_to_end(key)
        self.current_size += value_size

    def stats(self):
        """Return cache statistics"""
        return {
            'entries': len(self.cache),
            'size_mb': round(self.current_size / (1024 * 1024), 2),
            'max_size_mb': round(self.max_size_bytes / (1024 * 1024), 2)
        }


class MedicalSearchEngine:
    def __init__(self):
        print("🚀 Initializing HNSW Medical Image Search Engine (Bone Fractures)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📱 Device: {self.device}")

        print("📥 Loading BiomedCLIP model (this may take a moment)...")
        print("   Model: hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        
        # Load BiomedCLIP model using open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        self.tokenizer = open_clip.get_tokenizer(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        print("✅ BiomedCLIP model loaded successfully")

        # Initialize image cache
        cache_size_bytes = IMAGE_CACHE_SIZE_MB * 1024 * 1024
        self.image_cache = LRUImageCache(cache_size_bytes)
        print(f"💾 Initialized image cache: {IMAGE_CACHE_SIZE_MB}MB")

        # Initialize HTTP session with connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })

        self.index = None
        self.image_urls = None
        self.url_mapping = {}  # Mapping from incomplete URLs to complete URLs
        self.load_url_mapping()
        self.load_index()

    def load_url_mapping(self):
        """Load URL mapping from cloudinary_urls.json if available to fix incomplete URLs"""
        json_path = os.path.join(os.path.dirname(__file__), '..', 'medical_embedder', 'cloudinary_urls.json')
        if not os.path.exists(json_path):
            # Try alternative location
            json_path = os.path.join(os.path.dirname(__file__), 'cloudinary_urls.json')
        
        if os.path.exists(json_path):
            try:
                import json
                with open(json_path, 'r') as f:
                    cloudinary_data = json.load(f)
                
                # Create mapping from base URL (up to filename) to complete URL
                for item in cloudinary_data:
                    if 'url' in item:
                        complete_url = item['url']
                        # Extract base path (up to last slash) for mapping
                        if '/' in complete_url:
                            # Get the part before the filename
                            base_path = '/'.join(complete_url.split('/')[:-1]) + '/'
                            self.url_mapping[base_path] = complete_url
                            # Also map without trailing slash
                            base_path_no_slash = '/'.join(complete_url.split('/')[:-1])
                            self.url_mapping[base_path_no_slash] = complete_url
                
                print(f"✅ Loaded {len(self.url_mapping)} URL mappings from cloudinary_urls.json")
            except Exception as e:
                print(f"⚠️  Could not load URL mapping: {e}")
        else:
            print("ℹ️  cloudinary_urls.json not found - incomplete URLs may not be fixed")

    def fix_incomplete_url(self, url):
        """Fix incomplete URLs (ending with /) using URL mapping"""
        if not url or not isinstance(url, str):
            return url
        
        url = url.strip()
        # Check if URL ends with trailing slash (incomplete)
        if url.endswith('/'):
            # Try to find complete URL in mapping
            if url in self.url_mapping:
                fixed_url = self.url_mapping[url]
                print(f"🔧 Fixed incomplete URL: {url[:60]}... → {fixed_url[:60]}...")
                return fixed_url
            else:
                print(f"⚠️  Incomplete URL not found in mapping: {url[:80]}...")
        
        return url

    def load_index(self):
        """Load the pre-built HNSW index and image URLs"""
        try:
            # Determine the .bin filename
            index_filename = os.path.splitext(H5_FILE_PATH)[0] + ".bin"

            # Check if .h5 file exists (required for metadata and URLs)
            if not os.path.exists(H5_FILE_PATH):
                print(f"❌ Error: {H5_FILE_PATH} not found!")
                print("   Please ensure the h5 file is in the backend directory")
                print("   Run medical_embedder.ipynb to generate the file")
                return

            # Check if .bin file exists for faster index loading
            bin_exists = os.path.exists(index_filename)
            if bin_exists:
                print(f"⚡ Found existing HNSW index: {index_filename}")
            else:
                print(f"📂 .bin file not found, will build index from embeddings")

            print(f"📂 Loading medical image embeddings from HDF5 file: {H5_FILE_PATH}")

            # Load embeddings and URLs
            with h5py.File(H5_FILE_PATH, "r") as f:
                print(f"   Available datasets: {list(f.keys())}")

                # Support both 'image_path' (local) and 'urls' (online) for flexibility
                if 'image_path' in f:
                    self.image_urls = f["image_path"][:]
                    print(f"   → Loaded {len(self.image_urls)} image paths (local storage)")
                elif 'urls' in f:
                    self.image_urls = f["urls"][:]
                    print(f"   → Loaded {len(self.image_urls)} image URLs (online storage)")
                elif 'image_url' in f:
                    self.image_urls = f["image_url"][:]
                    print(f"   → Loaded {len(self.image_urls)} image URLs (online storage)")
                else:
                    raise ValueError("No path/URL dataset found in h5 file. Expected 'image_path', 'urls', or 'image_url'")

                embs = f["embeddings"][:]
                print(f"   → Loaded embeddings shape: {embs.shape}")

            # Sample a few URLs to verify format
            sample_urls = self.image_urls[:3]
            print("   Sample URLs:")
            for i, url in enumerate(sample_urls):
                decoded = url.decode('utf-8') if isinstance(url, bytes) else str(url)
                print(f"      {i+1}. {decoded[:80]}...")

            dim = embs.shape[1]

            # Check if .bin file exists for faster loading
            if os.path.exists(index_filename):
                print(f"⚡ Found existing HNSW index: {index_filename}")
                print(f"🔧 Loading HNSW index from .bin file (fast mode)...")
                self.index = hnswlib.Index(space='cosine', dim=dim)
                self.index.load_index(index_filename, max_elements=MAX_HNSW_ELEMENTS)
                self.index.set_ef(200)
                print(f"✅ HNSW index loaded successfully from .bin file!")
            else:
                # Build index from scratch if .bin doesn't exist
                print(f"🔧 Building HNSW index from scratch (capacity: {MAX_HNSW_ELEMENTS:,} images)...")
                print(f"   (This may take a while for the first run)")
                self.index = hnswlib.Index(space='cosine', dim=dim)
                self.index.init_index(max_elements=MAX_HNSW_ELEMENTS, ef_construction=400, M=200)
                self.index.set_ef(200)
                self.index.add_items(embs)
                # Save index for future runs
                self.index.save_index(index_filename)
                print(f"💾 Saved HNSW index to: {index_filename}")
                print(f"   (Next startup will be faster!)")

            print(f"✅ Medical search engine ready! Indexed {len(self.image_urls):,} X-ray images")
            print(f"📊 Index dimension: {dim}, Device: {self.device}")
            print(f"🔍 HNSW params: M=200, ef_construction=400, ef=200")

        except Exception as e:
            print(f"❌ Error loading index: {e}")
            import traceback
            traceback.print_exc()

    def fetch_image_from_url(self, url):
        """
        Fetch image from URL with robust error handling for multiple sources.
        Supports Cloudinary, Flickr, Pinterest, Google Images, etc.
        """
        # Check cache first
        # Note: MD5 is used here only for cache key generation (not for security)
        cache_key = hashlib.md5(url.encode()).hexdigest()  # nosec
        cached = self.image_cache.get(cache_key)
        if cached:
            return cached

        try:
            # Prepare headers based on URL domain
            headers = self.session.headers.copy()

            # Domain-specific headers for better compatibility
            if 'cloudinary.com' in url:
                # Cloudinary CDN - no special headers needed
                pass
            elif 'pinterest.com' in url or 'pinimg.com' in url:
                headers['Referer'] = 'https://www.pinterest.com/'
            elif 'reddit.com' in url or 'redd.it' in url:
                headers['Referer'] = 'https://www.reddit.com/'
            elif 'google.com' in url or 'googleusercontent.com' in url:
                headers['Referer'] = 'https://www.google.com/'

            # Fetch image with timeout
            response = self.session.get(
                url,
                headers=headers,
                timeout=IMAGE_FETCH_TIMEOUT,
                stream=True,
                allow_redirects=True
            )

            # Check if response is successful
            if response.status_code != 200:
                print(f"⚠️  Failed to fetch image (status {response.status_code}): {url[:100]}")
                return None

            # Verify content type
            content_type = response.headers.get('content-type', '').lower()
            if not content_type.startswith('image/'):
                print(f"⚠️  Invalid content type ({content_type}): {url[:100]}")
                return None

            # Read image data
            image_data = response.content

            # Validate it's actually an image by trying to open it
            try:
                Image.open(io.BytesIO(image_data)).verify()
            except Exception as e:
                print(f"⚠️  Invalid image data: {url[:100]} - {e}")
                return None

            # Cache the successful fetch
            self.image_cache.put(cache_key, image_data)

            return image_data

        except requests.exceptions.Timeout:
            print(f"⚠️  Timeout fetching image: {url[:100]}")
            return None
        except requests.exceptions.ConnectionError:
            print(f"⚠️  Connection error fetching image: {url[:100]}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Request error fetching image: {url[:100]} - {e}")
            return None
        except Exception as e:
            print(f"⚠️  Unexpected error fetching image: {url[:100]} - {e}")
            return None

    def prefetch_images_for_results(self, results):
        """
        Prefetch and encode images for search results.
        This reduces frontend requests at the cost of initial latency.
        """
        for result in results:
            url = result['url']

            # Fetch image
            image_data = self.fetch_image_from_url(url)

            if image_data:
                # Detect mime type
                url_lower = url.lower()
                if url_lower.endswith('.png'):
                    mime_type = 'image/png'
                elif url_lower.endswith('.gif'):
                    mime_type = 'image/gif'
                elif url_lower.endswith('.webp'):
                    mime_type = 'image/webp'
                elif url_lower.endswith('.bmp'):
                    mime_type = 'image/bmp'
                else:
                    mime_type = 'image/jpeg'

                # Encode to base64
                img_base64 = base64.b64encode(image_data).decode('utf-8')
                result['image_data'] = f"data:{mime_type};base64,{img_base64}"
            else:
                # Mark as unavailable
                result['image_data'] = None

        return results

    def text_to_vector(self, text):
        """Convert medical text query to vector using BiomedCLIP"""
        text_tokens = self.tokenizer([text.strip()]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().astype(np.float32)

    def search(self, query, k=20):
        """Search for similar medical images using text query"""
        if self.index is None:
            return []

        try:
            # Convert query to vector
            query_vector = self.text_to_vector(query)

            # Search in HNSW index
            indices, distances = self.index.knn_query(query_vector, k=k)

            # Get image URLs and scores
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                image_url = self.image_urls[idx].decode('utf-8') if isinstance(self.image_urls[idx], bytes) else str(self.image_urls[idx])
                image_url = image_url.strip()  # Remove any trailing whitespace/newlines
                # Fix incomplete URLs (ending with /)
                image_url = self.fix_incomplete_url(image_url)
                similarity_score = 1 - distance  # Convert distance to similarity
                results.append({
                    'path': image_url,  # Keep 'path' key for frontend compatibility
                    'url': image_url,   # Also provide 'url' key for clarity
                    'score': float(similarity_score)
                })

            return results
        except Exception as e:
            print(f"❌ Search error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def image_to_vector(self, image):
        """Convert PIL image (X-ray) to vector using BiomedCLIP"""
        try:
            # Preprocess image and convert to tensor
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            return image_features.cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"❌ Image encoding error: {e}")
            return None

    def search_by_image(self, image, k=20):
        """Search for similar medical images using uploaded X-ray image"""
        if self.index is None:
            return []

        try:
            # Convert image to vector
            query_vector = self.image_to_vector(image)
            if query_vector is None:
                return []

            # Search in HNSW index
            indices, distances = self.index.knn_query(query_vector, k=k)

            # Get image URLs and scores
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                image_url = self.image_urls[idx].decode('utf-8') if isinstance(self.image_urls[idx], bytes) else str(self.image_urls[idx])
                image_url = image_url.strip()  # Remove any trailing whitespace/newlines
                # Fix incomplete URLs (ending with /)
                image_url = self.fix_incomplete_url(image_url)
                similarity_score = 1 - distance  # Convert distance to similarity
                results.append({
                    'path': image_url,  # Keep 'path' key for frontend compatibility
                    'url': image_url,   # Also provide 'url' key for clarity
                    'score': float(similarity_score)
                })

            return results
        except Exception as e:
            print(f"❌ Image search error: {e}")
            import traceback
            traceback.print_exc()
            return []


# Initialize search engine
print("\n" + "="*70)
print("  HNSW MEDICAL IMAGE SEARCH BACKEND - BONE FRACTURES")
print("="*70 + "\n")

search_engine = MedicalSearchEngine()

print("\n" + "="*70)
print("  Server initialization complete!")
print("="*70 + "\n")


@app.route('/search', methods=['POST'])
def search_images():
    """API endpoint for text-based medical image search"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        k = data.get('k', 20)

        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400

        # Validate k is a positive integer
        try:
            k = int(k)
            if k <= 0:
                return jsonify({'error': 'k must be a positive integer'}), 400
        except (ValueError, TypeError):
            return jsonify({'error': 'k must be a valid integer'}), 400

        print(f"🔍 Medical text search: '{query}' (k={k})")

        # Perform search
        results = search_engine.search(query, k)

        # Optionally prefetch images
        if PREFETCH_IMAGES:
            print(f"   📥 Prefetching {len(results)} images...")
            results = search_engine.prefetch_images_for_results(results)
            print(f"   ✅ Prefetch complete")

        return jsonify({
            'query': query,
            'query_type': 'text',
            'results': results,
            'total': len(results),
            'prefetched': PREFETCH_IMAGES
        })

    except Exception as e:
        print(f"❌ Search error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/search/image', methods=['POST'])
def search_by_image():
    """API endpoint for X-ray image-based search"""
    try:
        # Check if image file is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400

        # Get and validate k parameter from form data
        k = request.form.get('k', 20)
        try:
            k = int(k)
            if k <= 0:
                return jsonify({'error': 'k must be a positive integer'}), 400
        except (ValueError, TypeError):
            return jsonify({'error': 'k must be a valid integer'}), 400

        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''

        if file_ext not in allowed_extensions:
            return jsonify({'error': 'Invalid file type. Supported formats: PNG, JPG, JPEG, GIF, BMP, WEBP'}), 400

        print(f"🖼️  Medical image search: {file.filename} (k={k})")

        # Open and process the image
        try:
            image = Image.open(file.stream).convert('RGB')
        except Exception as e:
            return jsonify({'error': f'Cannot process image: {str(e)}'}), 400

        # Perform search
        results = search_engine.search_by_image(image, k)

        # Optionally prefetch images
        if PREFETCH_IMAGES:
            print(f"   📥 Prefetching {len(results)} images...")
            results = search_engine.prefetch_images_for_results(results)
            print(f"   ✅ Prefetch complete")

        return jsonify({
            'query': file.filename,
            'query_type': 'image',
            'results': results,
            'total': len(results),
            'prefetched': PREFETCH_IMAGES
        })

    except Exception as e:
        print(f"❌ Image search error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/image', methods=['GET'])
def serve_image():
    """Serve images from the local filesystem using query parameter"""
    try:
        # Get path from query parameter
        image_path = request.args.get('path', '').strip()
        
        if not image_path:
            return jsonify({'error': 'No path provided'}), 400
        
        # Security: ensure path is within allowed directories
        if '..' in image_path:
            return jsonify({'error': 'Invalid path'}), 400
        
        # If it's an absolute path, use it directly
        if os.path.isabs(image_path):
            full_path = image_path
        else:
            # Get the directory where server_medical.py is located (backend/)
            base_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Handle relative paths (e.g., "./images/123.jpg" or "images/123.jpg")
            if image_path.startswith('./'):
                image_path = image_path[2:]  # Remove "./" prefix
            
            # Construct absolute path
            full_path = os.path.join(base_dir, image_path)
            
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


@app.route('/image-proxy', methods=['GET'])
def serve_image_proxy():
    """
    Proxy endpoint to fetch images from external URLs (Cloudinary, etc.) - DEPRECATED.
    Use /image/<path> for local images instead.
    """
    try:
        # Get URL from query parameter
        image_url = request.args.get('url', '').strip()

        if not image_url:
            return jsonify({'error': 'URL parameter is required'}), 400

        # Validate URL format
        if not image_url.startswith(('http://', 'https://')):
            return jsonify({'error': 'Invalid URL format. Must start with http:// or https://'}), 400

        # Fetch image from URL
        image_data = search_engine.fetch_image_from_url(image_url)

        if image_data is None:
            return jsonify({
                'error': 'Failed to fetch image',
                'url': image_url,
                'message': 'Could not retrieve image from the provided URL. It may be unavailable or access restricted.'
            }), 404

        # Detect mime type from URL extension or default to jpeg
        url_lower = image_url.lower()
        if url_lower.endswith('.png'):
            mime_type = 'image/png'
        elif url_lower.endswith('.gif'):
            mime_type = 'image/gif'
        elif url_lower.endswith('.webp'):
            mime_type = 'image/webp'
        elif url_lower.endswith('.bmp'):
            mime_type = 'image/bmp'
        else:
            mime_type = 'image/jpeg'

        # Encode to base64 for frontend
        img_base64 = base64.b64encode(image_data).decode('utf-8')

        return jsonify({
            'image_data': f"data:{mime_type};base64,{img_base64}",
            'url': image_url,
            'source': 'proxy'
        })

    except Exception as e:
        print(f"❌ Image proxy error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Server error',
            'message': str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with detailed status"""
    cache_stats = search_engine.image_cache.stats()

    return jsonify({
        'status': 'healthy',
        'version': '1.0-medical',
        'service': 'medical-bone-fracture-search',
        'model': 'BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        'model_loaded': search_engine.index is not None,
        'device': search_engine.device,
        'total_images': len(search_engine.image_urls) if search_engine.image_urls is not None else 0,
        'cache': cache_stats,
        'config': {
            'h5_file': H5_FILE_PATH,
            'cache_size_mb': IMAGE_CACHE_SIZE_MB,
            'fetch_timeout': IMAGE_FETCH_TIMEOUT,
            'max_hnsw_elements': MAX_HNSW_ELEMENTS,
            'prefetch_images': PREFETCH_IMAGES
        }
    })


@app.route('/cache/stats', methods=['GET'])
def cache_stats():
    """Get detailed cache statistics"""
    stats = search_engine.image_cache.stats()
    return jsonify(stats)


@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear the image cache (useful for development/debugging)"""
    search_engine.image_cache = LRUImageCache(IMAGE_CACHE_SIZE_MB * 1024 * 1024)
    return jsonify({
        'message': 'Cache cleared successfully',
        'cache_size_mb': IMAGE_CACHE_SIZE_MB
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))  # Port 5002 for medical search
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'

    print(f"🌐 Starting Flask Medical Search Server on http://0.0.0.0:{port}")
    print(f"🔧 Debug mode: {debug}")
    print(f"🌍 CORS: Enabled for all origins (suitable for Vercel deployment)")
    print(f"💾 Image cache: {IMAGE_CACHE_SIZE_MB}MB")
    print(f"⏱️  Fetch timeout: {IMAGE_FETCH_TIMEOUT}s")
    print(f"📁 H5 file: {H5_FILE_PATH}")
    print(f"📊 Max capacity: {MAX_HNSW_ELEMENTS:,} images")
    print(f"🚀 Prefetch mode: {'ENABLED' if PREFETCH_IMAGES else 'DISABLED'}")
    print(f"\n📡 API Endpoints:")
    print(f"   - POST /search              (medical text search)")
    print(f"   - POST /search/image        (X-ray image search)")
    print(f"   - GET  /image-proxy?url=... (image proxy)")
    print(f"   - GET  /health              (health check)")
    print(f"   - GET  /cache/stats         (cache statistics)")
    print(f"   - POST /cache/clear         (clear cache)")
    print(f"\n✅ Access the API at: http://localhost:{port}/health")
    print(f"\n💡 Example queries:")
    print(f"   - 'broken leg bone'")
    print(f"   - 'distal radius fracture'")
    print(f"   - 'comminuted femur fracture'")
    print(f"   - 'osteoporotic hip fracture'")
    print("\nPress Ctrl+C to stop the server\n")

    # Note: host='0.0.0.0' is intentional for deployment (VPS, Docker, etc.)
    # For local development only, use host='127.0.0.1' instead
    app.run(host='0.0.0.0', port=port, debug=debug)  # nosec

