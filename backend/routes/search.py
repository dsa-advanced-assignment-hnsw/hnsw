"""
Search route blueprint.
Handles text and image-based search endpoints.
"""

from flask import Blueprint, request, jsonify
from PIL import Image

search_bp = Blueprint('search', __name__)


def _validate_k_parameter(k):
    """
    Validate k parameter for search.
    
    Args:
        k: Number of results to return
        
    Returns:
        tuple: (is_valid, error_message, validated_k)
    """
    try:
        if k is None:
            k = 20
        k = int(k)
        if k <= 0:
            return False, 'k must be a positive integer', None
        if k > 100:
            k = 100  # Cap at 100
        return True, '', k
    except (ValueError, TypeError):
        return False, 'k must be a valid integer', None


@search_bp.route('/search', methods=['POST'])
def search_text():
    """Text-based search endpoint."""
    try:
        engine = search_bp.engine
        data = request.get_json()
        query = data.get('query', '').strip()
        k = data.get('k', 20)
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        # Validate k
        is_valid, error_msg, k = _validate_k_parameter(k)
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        
        print(f"🔍 Text search: '{query}' (k={k})")
        
        # Perform search
        results = engine.search(query, k)
        
        return jsonify({
            'query': query,
            'query_type': 'text',
            'results': results,
            'total': len(results)
        })
    
    except Exception as e:
        print(f"❌ Search error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@search_bp.route('/search/image', methods=['POST'])
def search_image():
    """Image-based search endpoint."""
    try:
        engine = search_bp.engine
        
        # Check if image file is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Get and validate k parameter
        k = request.form.get('k', 20)
        is_valid, error_msg, k = _validate_k_parameter(k)
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
        if file_ext not in allowed_extensions:
            return jsonify({'error': 'Invalid file type. Supported: PNG, JPG, JPEG, GIF, BMP, WEBP'}), 400
        
        print(f"🖼️  Image search: {file.filename} (k={k})")
        
        # Open and process the image
        try:
            image = Image.open(file.stream).convert('RGB')
        except Exception as e:
            return jsonify({'error': f'Cannot process image: {str(e)}'}), 400
        
        # Perform search (if engine supports image search)
        if not hasattr(engine, 'search_by_image'):
            return jsonify({'error': 'Image search not supported for this engine'}), 400
        
        results = engine.search_by_image(image, k)
        
        return jsonify({
            'query': file.filename,
            'query_type': 'image',
            'results': results,
            'total': len(results)
        })
    
    except Exception as e:
        print(f"❌ Image search error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@search_bp.route('/search/file', methods=['POST'])
def search_file():
    """File-based search endpoint (for papers)."""
    try:
        engine = search_bp.engine
        
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get and validate k parameter
        k = request.form.get('k', 20)
        is_valid, error_msg, k = _validate_k_parameter(k)
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        
        # Validate file type
        allowed_extensions = {'txt', 'pdf', 'md'}
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
        if file_ext not in allowed_extensions:
            return jsonify({'error': 'Invalid file type. Supported: .txt, .pdf, .md'}), 400
        
        print(f"📄 File search: {file.filename} (k={k})")
        
        # Read file content
        try:
            # For now, extract text from file
            if file_ext == 'txt':
                text = file.read().decode('utf-8')
            elif file_ext == 'md':
                text = file.read().decode('utf-8')
            elif file_ext == 'pdf':
                # TODO: Add PDF text extraction
                return jsonify({'error': 'PDF parsing not yet implemented'}), 501
            else:
                return jsonify({'error': 'Unsupported file type'}), 400
        except Exception as e:
            return jsonify({'error': f'Cannot read file: {str(e)}'}), 400
        
        # Perform search using file text
        results = engine.search(text, k)
        
        return jsonify({
            'query': file.filename,
            'query_type': 'file',
            'results': results,
            'total': len(results)
        })
    
    except Exception as e:
        print(f"❌ File search error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
