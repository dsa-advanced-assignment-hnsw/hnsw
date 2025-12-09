"""
Image proxy route blueprint.
Handles serving images from local files and proxying remote URLs.
"""

from flask import Blueprint, request, jsonify

image_proxy_bp = Blueprint('image_proxy', __name__)


@image_proxy_bp.route('/image/<path:image_path>')
def serve_local_image(image_path):
    """Serve images from local dataset."""
    try:
        fetcher = image_proxy_bp.image_fetcher
        base_dir = image_proxy_bp.base_dir
        
        # Fetch from local file system
        image_data = fetcher.fetch_from_local(image_path, base_dir)
        
        if image_data is None:
            return jsonify({
                'error': 'Image not found',
                'path': image_path,
                'message': 'The image file does not exist or cannot be read'
            }), 404
        
        # Convert to base64
        data_uri = fetcher.to_base64(image_data, image_path)
        
        return jsonify({
            'image_data': data_uri,
            'path': image_path
        })
    
    except Exception as e:
        print(f"❌ Error serving image: {e}")
        return jsonify({
            'error': 'Server error',
            'path': image_path,
            'message': str(e)
        }), 500


@image_proxy_bp.route('/image-proxy', methods=['GET'])
def proxy_remote_image():
    """Proxy endpoint to fetch images from external URLs."""
    try:
        fetcher = image_proxy_bp.image_fetcher
        
        # Get URL from query parameter
        image_url = request.args.get('url', '').strip()
        
        if not image_url:
            return jsonify({'error': 'URL parameter is required'}), 400
        
        # Validate URL format
        if not image_url.startswith(('http://', 'https://')):
            return jsonify({'error': 'Invalid URL format'}), 400
        
        # Fetch image from URL
        image_data = fetcher.fetch_from_url(image_url)
        
        if image_data is None:
            return jsonify({
                'error': 'Failed to fetch image',
                'url': image_url,
                'message': 'Image unavailable or access restricted'
            }), 404
        
        # Convert to base64
        data_uri = fetcher.to_base64(image_data, image_url)
        
        return jsonify({
            'image_data': data_uri,
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
