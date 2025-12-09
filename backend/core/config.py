"""
Configuration management for HNSW Search Engine.
Centralized environment variable handling and defaults.
"""

import os


class Config:
    """
    Application configuration loaded from environment variables with sensible defaults.
    """
    
    # Image cache settings
    IMAGE_CACHE_SIZE_MB = int(os.environ.get('IMAGE_CACHE_SIZE_MB', 100))
    IMAGE_FETCH_TIMEOUT = int(os.environ.get('IMAGE_FETCH_TIMEOUT', 10))
    
    # HNSW index settings
    HNSW_M = int(os.environ.get('HNSW_M', 200))
    HNSW_EF_CONSTRUCTION = int(os.environ.get('HNSW_EF_CONSTRUCTION', 400))
    HNSW_EF_SEARCH = int(os.environ.get('HNSW_EF_SEARCH', 200))
    MAX_HNSW_ELEMENTS = int(os.environ.get('MAX_HNSW_ELEMENTS', 2000000))
    
    # File paths
    IMAGE_H5_PATH = os.environ.get('IMAGE_H5_PATH', 'Images_Embedbed_0-100000.h5')
    PAPER_H5_PATH = os.environ.get('PAPER_H5_PATH', 'Papers_Embedbed_0-100000.h5')
    MEDICAL_H5_PATH = os.environ.get('MEDICAL_H5_PATH', 'Medical_Fractures_Embedbed.h5')
    
    # Feature flags
    PREFETCH_IMAGES = os.environ.get('PREFETCH_IMAGES', 'false').lower() == 'true'
    
    # Flask settings
    FLASK_DEBUG = os.environ.get('FLASK_DEBUG', '0') == '1'
    PORT = int(os.environ.get('PORT', 5000))
    
    # CORS settings
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')
    
    @classmethod
    def validate(cls):
        """
        Validate configuration values.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if cls.IMAGE_CACHE_SIZE_MB <= 0:
            raise ValueError("IMAGE_CACHE_SIZE_MB must be positive")
        
        if cls.HNSW_M <= 0:
            raise ValueError("HNSW_M must be positive")
        
        if cls.HNSW_EF_CONSTRUCTION <= 0:
            raise ValueError("HNSW_EF_CONSTRUCTION must be positive")
        
        if cls.HNSW_EF_SEARCH <= 0:
            raise ValueError("HNSW_EF_SEARCH must be positive")
    
    @classmethod
    def print_config(cls):
        """Print current configuration to console."""
        print("=" * 70)
        print("  CONFIGURATION")
        print("=" * 70)
        print(f"📁 File Paths:")
        print(f"   - Image H5:   {cls.IMAGE_H5_PATH}")
        print(f"   - Paper H5:   {cls.PAPER_H5_PATH}")
        print(f"   - Medical H5: {cls.MEDICAL_H5_PATH}")
        print(f"")
        print(f"🔧 HNSW Parameters:")
        print(f"   - M:                   {cls.HNSW_M}")
        print(f"   - ef_construction:     {cls.HNSW_EF_CONSTRUCTION}")
        print(f"   - ef_search:           {cls.HNSW_EF_SEARCH}")
        print(f"   - max_elements:        {cls.MAX_HNSW_ELEMENTS:,}")
        print(f"")
        print(f"💾 Cache Settings:")
        print(f"   - Size:                {cls.IMAGE_CACHE_SIZE_MB}MB")
        print(f"   - Fetch timeout:       {cls.IMAGE_FETCH_TIMEOUT}s")
        print(f"   - Prefetch images:     {cls.PREFETCH_IMAGES}")
        print(f"")
        print(f"🌐 Server Settings:")
        print(f"   - Port:                {cls.PORT}")
        print(f"   - Debug:               {cls.FLASK_DEBUG}")
        print(f"   - CORS origins:        {', '.join(cls.CORS_ORIGINS)}")
        print("=" * 70)
