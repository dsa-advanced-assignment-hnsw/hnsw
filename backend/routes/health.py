"""
Health and cache route blueprint.
Provides health check and cache management endpoints.
"""

from flask import Blueprint, jsonify

health_bp = Blueprint('health', __name__)


@health_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with detailed status."""
    try:
        engine = health_bp.engine
        cache = health_bp.cache
        
        # Get stats
        engine_stats = engine.get_stats()
        cache_stats = cache.stats() if cache else {'entries': 0, 'size_mb': 0}
        
        return jsonify({
            'status': 'healthy',
            'version': '3.0',
            **engine_stats,
            'cache': cache_stats
        })
    
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@health_bp.route('/cache/stats', methods=['GET'])
def cache_stats():
    """Get detailed cache statistics."""
    try:
        cache = health_bp.cache
        
        if cache is None:
            return jsonify({'message': 'Cache not enabled'}), 200
        
        return jsonify(cache.stats())
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@health_bp.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear the image cache."""
    try:
        cache = health_bp.cache
        
        if cache is None:
            return jsonify({'message': 'Cache not enabled'}), 200
        
        cache.clear()
        
        return jsonify({
            'message': 'Cache cleared successfully',
            'stats': cache.stats()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
