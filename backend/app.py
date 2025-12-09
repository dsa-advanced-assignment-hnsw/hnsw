"""
Flask application factory for HNSW Search Engine.
Creates and configures Flask app with routes and engine.
"""

import os
from flask import Flask
from flask_cors import CORS

from core.config import Config
from core.cache import LRUImageCache
from core.image_fetcher import ImageFetcher
from engines.image_engine import ImageSearchEngine
from engines.paper_engine import PaperSearchEngine
from engines.medical_engine import MedicalSearchEngine
from routes.search import search_bp
from routes.image_proxy import image_proxy_bp
from routes.health import health_bp


def create_app(engine_type='image'):
    """
    Create and configure Flask application.
    
    Args:
        engine_type: Type of search engine ('image', 'paper', or 'medical')
        
    Returns:
        Flask application instance
    """
    # Validate configuration
    Config.validate()
    
    # Print configuration
    print("\n" + "="*70)
    print(f"  HNSW SEARCH ENGINE v3.0 - {engine_type.upper()}")
    print("="*70 + "\n")
    Config.print_config()
    print()
    
    # Create Flask app
    app = Flask(__name__)
    
    # Configure CORS
    CORS(app,
         origins=Config.CORS_ORIGINS,
         allow_headers=['Content-Type', 'ngrok-skip-browser-warning', 'Authorization'],
         methods=['GET', 'POST', 'OPTIONS'],
         supports_credentials=False)
    
    # Initialize cache
    cache_size_bytes = Config.IMAGE_CACHE_SIZE_MB * 1024 * 1024
    cache = LRUImageCache(cache_size_bytes)
    print(f"💾 Initialized image cache: {Config.IMAGE_CACHE_SIZE_MB}MB\n")
    
    # Initialize image fetcher
    image_fetcher = ImageFetcher(cache)
    
    # Initialize search engine based on type
    print(f"🚀 Initializing {engine_type.upper()} search engine...\n")
    if engine_type == 'image':
        engine = ImageSearchEngine()
    elif engine_type == 'paper':
        engine = PaperSearchEngine()
    elif engine_type == 'medical':
        engine = MedicalSearchEngine()
    else:
        raise ValueError(f"Unknown engine type: {engine_type}")
    
    # Attach engine and cache to blueprints
    search_bp.engine = engine
    image_proxy_bp.image_fetcher = image_fetcher
    image_proxy_bp.base_dir = os.path.dirname(os.path.abspath(__file__))
    health_bp.engine = engine
    health_bp.cache = cache
    
    # Register blueprints
    app.register_blueprint(search_bp)
    app.register_blueprint(image_proxy_bp)
    app.register_blueprint(health_bp)
    
    print("\n" + "="*70)
    print("  SERVER INITIALIZATION COMPLETE")
    print("="*70)
    print(f"\n📡 API Endpoints:")
    print(f"   - POST /search              (text search)")
    print(f"   - POST /search/image        (image search)")
    print(f"   - POST /search/file         (file search)")
    print(f"   - GET  /image/<path>        (serve local image)")
    print(f"   - GET  /image-proxy?url=... (proxy remote image)")
    print(f"   - GET  /health              (health check)")
    print(f"   - GET  /cache/stats         (cache statistics)")
    print(f"   - POST /cache/clear         (clear cache)")
    print()
    
    return app


if __name__ == '__main__':
    # Determine engine type from environment or default to image
    engine_type = os.environ.get('ENGINE_TYPE', 'image').lower()
    
    # Create app
    app = create_app(engine_type)
    
    # Run server
    print(f"🌐 Starting Flask server on http://0.0.0.0:{Config.PORT}")
    print(f"🔧 Debug mode: {Config.FLASK_DEBUG}")
    print(f"🌍 CORS origins: {', '.join(Config.CORS_ORIGINS)}")
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(
        host='0.0.0.0',
        port=Config.PORT,
        debug=Config.FLASK_DEBUG
    )
