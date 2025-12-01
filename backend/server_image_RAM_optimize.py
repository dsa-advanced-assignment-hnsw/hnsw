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
import requests
from collections import OrderedDict
import hashlib
import atexit # Dùng để đăng ký hàm đóng file
import sys

# ==============================================================================
# 1. FLASK SETUP & CONFIGURATION
# ==============================================================================

app = Flask(__name__)

# Enable CORS for flexible deployment
CORS(app,
     origins=['*'],
     allow_headers=['Content-Type', 'ngrok-skip-browser-warning', 'Authorization'],
     methods=['GET', 'POST', 'OPTIONS'],
     supports_credentials=False)

# Configuration from environment variables
IMAGE_CACHE_SIZE_MB = int(os.environ.get('IMAGE_CACHE_SIZE_MB', 100))
IMAGE_FETCH_TIMEOUT = int(os.environ.get('IMAGE_FETCH_TIMEOUT', 10))
H5_FILE_PATH = os.environ.get('H5_FILE_PATH', 'Image_Embedded.h5')
BIN_FILE_PATH = 'hnsw_image_index.bin'
MAX_HNSW_ELEMENTS = int(os.environ.get('MAX_HNSW_ELEMENTS', 2000000))
PREFETCH_IMAGES = os.environ.get('PREFETCH_IMAGES', 'false').lower() == 'true'

# ==============================================================================
# 2. CACHE CLASS (Giữ nguyên)
# ==============================================================================

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
        if value_size > self.max_size_bytes:
            return

        while self.current_size + value_size > self.max_size_bytes and self.cache:
            oldest_key, oldest_value = self.cache.popitem(last=False)
            self.current_size -= len(oldest_value)

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

# ==============================================================================
# 3. SEARCH ENGINE CLASS (Chế độ Đọc Trực Tiếp từ Disk)
# ==============================================================================

class SearchEngine:
    def __init__(self):
        print("🚀 Initializing HNSW Image Search Engine (Lazy Loading Mode)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📱 Device: {self.device}")

        print("📥 Loading CLIP model (this may take a moment)...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        print("✅ CLIP model loaded successfully")

        cache_size_bytes = IMAGE_CACHE_SIZE_MB * 1024 * 1024
        self.image_cache = LRUImageCache(cache_size_bytes)
        print(f"💾 Initialized image cache: {IMAGE_CACHE_SIZE_MB}MB")

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })

        self.index = None
        self.h5_file = None # Kết nối HDF5 sẽ được giữ mở
        self.dim = None
        self.max_elements = 0
        self.load_index()
        
        # ✅ Đăng ký hàm đóng file HDF5 khi server tắt
        atexit.register(self.close)

    def close(self):
        """Đảm bảo file HDF5 được đóng"""
        if self.h5_file:
            print(f"🔒 Closing HDF5 file: {H5_FILE_PATH}")
            self.h5_file.close()
            self.h5_file = None

    def __del__(self):
        """Dọn dẹp khi đối tượng bị hủy"""
        self.close()

    def load_index(self):
        """Load the pre-built HNSW index and keep HDF5 file open for direct access"""
        try:
            index_filename = BIN_FILE_PATH

            if not os.path.exists(H5_FILE_PATH):
                print(f"❌ Error: {H5_FILE_PATH} not found!")
                print("   Please ensure the h5 file is in the backend directory")
                return

            bin_exists = os.path.exists(index_filename)
            if bin_exists:
                print(f"⚡ Found existing HNSW index: {index_filename}")
            else:
                print(f"📂 .bin file not found, will build index from embeddings")

            print(f"📂 Opening HDF5 file (Lazy Loading Mode): {H5_FILE_PATH}")

            # ✅ Khắc phục: Giữ kết nối file h5py.File đang mở
            self.h5_file = h5py.File(H5_FILE_PATH, "r")

            # Lấy thông tin từ file HDF5 (Chỉ lấy metadata)
            self.dim = self.h5_file['embeddings'].shape[1]
            self.max_elements = len(self.h5_file['urls'])

            print(f"   → Loaded embeddings shape: {self.h5_file['embeddings'].shape}")
            print(f"   → URLs will be read from disk during search.")

            # Sample a few URLs to verify format
            sample_urls = self.h5_file['urls'][:3]
            print("   Sample URLs:")
            for i, url in enumerate(sample_urls):
                decoded = url.decode('utf-8') if isinstance(url, bytes) else str(url)
                print(f"      {i+1}. {decoded[:80]}...")

            # Check if .bin file exists for faster loading
            if os.path.exists(index_filename):
                print(f"⚡ Found existing HNSW index: {index_filename}")
                print(f"🔧 Loading HNSW index from .bin file (fast mode)...")
                self.index = hnswlib.Index(space='cosine', dim=self.dim)
                self.index.load_index(index_filename, max_elements=self.max_elements)
                self.index.set_ef(200)
                print(f"✅ HNSW index loaded successfully from .bin file!")
            else:
                # Build index from scratch if .bin doesn't exist
                print(f"🔧 Building HNSW index from scratch (capacity: {MAX_HNSW_ELEMENTS:,} images)...")
                print(f"   (Loading embeddings into RAM temporarily...)")
                
                # Cần tải embeddings vào RAM tạm thời để build index
                embeddings_data = self.h5_file['embeddings'][:] 
                
                self.index = hnswlib.Index(space='cosine', dim=self.dim)
                self.index.init_index(max_elements=MAX_HNSW_ELEMENTS, ef_construction=400, M=200)
                self.index.set_ef(200)
                self.index.add_items(embeddings_data)
                self.index.save_index(index_filename)
                print(f"💾 Saved HNSW index to: {index_filename}")
                del embeddings_data # Giải phóng RAM sau khi build
                
            print(f"✅ Search engine ready! Indexed {self.max_elements:,} images")
            print(f"📊 Index dimension: {self.dim}, Device: {self.device}")

        except Exception as e:
            print(f"❌ Error loading index: {e}")
            import traceback
            traceback.print_exc()
            self.close() # Đóng file nếu có lỗi

    # ... (Các hàm fetch_image_from_url, prefetch_images_for_results, text_to_vector giữ nguyên) ...

    def fetch_image_from_url(self, url):
        """Fetch image from URL with robust error handling for multiple sources."""
        # ... (Giữ nguyên) ...
        cache_key = hashlib.md5(url.encode()).hexdigest()
        cached = self.image_cache.get(cache_key)
        if cached:
            return cached

        try:
            headers = self.session.headers.copy()
            if 'pinterest.com' in url or 'pinimg.com' in url:
                headers['Referer'] = 'https://www.pinterest.com/'
            elif 'reddit.com' in url or 'redd.it' in url:
                headers['Referer'] = 'https://www.reddit.com/'
            elif 'google.com' in url or 'googleusercontent.com' in url:
                headers['Referer'] = 'https://www.google.com/'
            elif 'facebook.com' in url or 'fbcdn.net' in url:
                headers['Referer'] = 'https://www.facebook.com/'
            elif 'instagram.com' in url or 'cdninstagram.com' in url:
                headers['Referer'] = 'https://www.instagram.com/'

            response = self.session.get(
                url,
                headers=headers,
                timeout=IMAGE_FETCH_TIMEOUT,
                stream=True,
                allow_redirects=True
            )

            if response.status_code != 200:
                print(f"⚠️  Failed to fetch image (status {response.status_code}): {url[:100]}")
                return None

            content_type = response.headers.get('content-type', '').lower()
            if not content_type.startswith('image/'):
                print(f"⚠️  Invalid content type ({content_type}): {url[:100]}")
                return None

            image_data = response.content

            try:
                Image.open(io.BytesIO(image_data)).verify()
            except Exception as e:
                print(f"⚠️  Invalid image data: {url[:100]} - {e}")
                return None

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
        """Prefetch and encode images for search results."""
        # ... (Giữ nguyên) ...
        for result in results:
            url = result['url']

            image_data = self.fetch_image_from_url(url)

            if image_data:
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

                img_base64 = base64.b64encode(image_data).decode('utf-8')
                result['image_data'] = f"data:{mime_type};base64,{img_base64}"
            else:
                result['image_data'] = None

        return results

    def text_to_vector(self, text):
        """Convert text query to vector"""
        text_tokens = clip.tokenize([text.strip()]).to(self.device)
        with torch.no_grad():
            vector_text = self.model.encode_text(text_tokens)
            vector_text = vector_text / vector_text.norm(dim=-1, keepdim=True)
        return vector_text.cpu().numpy().astype(np.float32)

    def search(self, query, k=20):
        """Search for similar images using text query"""
        # Kiểm tra index và file HDF5
        if self.index is None or self.h5_file is None:
            return []

        try:
            query_vector = self.text_to_vector(query)
            indices, distances = self.index.knn_query(query_vector, k=k)

            results = []
            for idx, distance in zip(indices[0], distances[0]):
                # ✅ ĐỌC TRỰC TIẾP TỪ DISK
                url_bytes = self.h5_file['urls'][idx]
                image_url = url_bytes.decode('utf-8') if isinstance(url_bytes, bytes) else str(url_bytes)
                
                image_url = image_url.strip()
                similarity_score = 1 - distance
                results.append({
                    'path': image_url,
                    'url': image_url,
                    'score': float(similarity_score)
                })

            return results
        except Exception as e:
            print(f"❌ Search error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def image_to_vector(self, image):
        """Convert PIL image to vector"""
        try:
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
        if self.index is None or self.h5_file is None:
            return []

        try:
            query_vector = self.image_to_vector(image)
            if query_vector is None:
                return []

            indices, distances = self.index.knn_query(query_vector, k=k)

            results = []
            for idx, distance in zip(indices[0], distances[0]):
                # ✅ ĐỌC TRỰC TIẾP TỪ DISK
                url_bytes = self.h5_file['urls'][idx]
                image_url = url_bytes.decode('utf-8') if isinstance(url_bytes, bytes) else str(url_bytes)
                
                image_url = image_url.strip()
                similarity_score = 1 - distance
                results.append({
                    'path': image_url,
                    'url': image_url,
                    'score': float(similarity_score)
                })

            return results
        except Exception as e:
            print(f"❌ Image search error: {e}")
            import traceback
            traceback.print_exc()
            return []


# ==============================================================================
# 4. INITIALIZATION
# ==============================================================================

print("\n" + "="*70)
print("  HNSW IMAGE SEARCH BACKEND v2.0 - ONLINE IMAGES")
print("="*70 + "\n")

search_engine = SearchEngine()

print("\n" + "="*70)
print("  Server initialization complete!")
print("="*70 + "\n")


# ==============================================================================
# 5. FLASK ROUTES
# ==============================================================================

@app.route('/search', methods=['POST'])
def search_images():
    """API endpoint for text-based image search"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        k = data.get('k', 20)

        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400

        try:
            k = int(k)
            if k <= 0:
                return jsonify({'error': 'k must be a positive integer'}), 400
        except (ValueError, TypeError):
            return jsonify({'error': 'k must be a valid integer'}), 400

        print(f"🔍 Text search: '{query}' (k={k})")

        results = search_engine.search(query, k)

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
    """API endpoint for image-based search"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400

        k = request.form.get('k', 20)
        try:
            k = int(k)
            if k <= 0:
                return jsonify({'error': 'k must be a positive integer'}), 400
        except (ValueError, TypeError):
            return jsonify({'error': 'k must be a valid integer'}), 400

        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''

        if file_ext not in allowed_extensions:
            return jsonify({'error': 'Invalid file type. Supported formats: PNG, JPG, JPEG, GIF, BMP, WEBP'}), 400

        print(f"🖼️  Image search: {file.filename} (k={k})")

        try:
            image = Image.open(file.stream).convert('RGB')
        except Exception as e:
            return jsonify({'error': f'Cannot process image: {str(e)}'}), 400

        results = search_engine.search_by_image(image, k)

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


@app.route('/image-proxy', methods=['GET'])
def serve_image_proxy():
    """Proxy endpoint to fetch images from external URLs."""
    try:
        image_url = request.args.get('url', '').strip()

        if not image_url:
            return jsonify({'error': 'URL parameter is required'}), 400

        if not image_url.startswith(('http://', 'https://')):
            return jsonify({'error': 'Invalid URL format. Must start with http:// or https://'}), 400

        image_data = search_engine.fetch_image_from_url(image_url)

        if image_data is None:
            return jsonify({
                'error': 'Failed to fetch image',
                'url': image_url,
                'message': 'Could not retrieve image from the provided URL. It may be unavailable or access restricted.'
            }), 404

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
    total_images = search_engine.max_elements
    embedding_dim = search_engine.dim

    return jsonify({
        'status': 'healthy',
        'version': '2.0',
        'model_loaded': search_engine.index is not None,
        'device': search_engine.device,
        'total_images': total_images,
        'embedding_dim': embedding_dim if embedding_dim else 'N/A',
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
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'

    print(f"🌐 Starting Flask server v2.0 on http://0.0.0.0:{port}")
    print(f"🔧 Debug mode: {debug}")
    print(f"🌍 CORS: Enabled for all origins (suitable for Vercel deployment)")
    print(f"💾 Image cache: {IMAGE_CACHE_SIZE_MB}MB")
    print(f"⏱️  Fetch timeout: {IMAGE_FETCH_TIMEOUT}s")
    print(f"📁 H5 file: {H5_FILE_PATH}")
    print(f"📊 Max capacity: {MAX_HNSW_ELEMENTS:,} images")
    print(f"🚀 Prefetch mode: {'ENABLED' if PREFETCH_IMAGES else 'DISABLED'}")
    if PREFETCH_IMAGES:
        print(f"   💡 Images will be fetched server-side (reduces ngrok requests)")
    else:
        print(f"   💡 Images will be fetched client-side (use for production)")
    print(f"\n📡 API Endpoints:")
    print(f"   - POST /search              (text search)")
    print(f"   - POST /search/image        (image search)")
    print(f"   - GET  /image-proxy?url=... (image proxy)")
    print(f"   - GET  /health              (health check)")
    print(f"   - GET  /cache/stats         (cache statistics)")
    print(f"   - POST /cache/clear         (clear cache)")
    print(f"\n✅ Access the API at: http://localhost:{port}/health")
    if not PREFETCH_IMAGES:
        print(f"💡 Tip: Set PREFETCH_IMAGES=true to reduce ngrok requests")
    print("\nPress Ctrl+C to stop the server\n")
    print(f"Total images indexed: {search_engine.max_elements:,}") 

    app.run(host='0.0.0.0', port=port, debug=debug)