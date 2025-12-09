"""
Centralized image fetching and proxying logic.
Handles both local file paths and remote URLs.
"""

import os
import base64
import io
import requests
from PIL import Image

from core.cache import LRUImageCache
from core.config import Config


class ImageFetcher:
    """
    Handles fetching images from various sources (local files, URLs, Cloudinary, etc.)
    with caching and robust error handling.
    """
    
    def __init__(self, cache: LRUImageCache = None):
        """
        Initialize image fetcher.
        
        Args:
            cache: Optional LRUImageCache instance. If None, caching is disabled.
        """
        self.cache = cache
        
        # Initialize HTTP session with connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_from_url(self, url):
        """
        Fetch image from URL with robust error handling for multiple sources.
        Supports Flickr, Pinterest, Google, Cloudinary, etc.
        
        Args:
            url: Image URL
            
        Returns:
            bytes: Image data or None if fetch failed
        """
        # Check cache first
        if self.cache:
            cache_key = LRUImageCache.make_cache_key(url)
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        
        try:
            # Prepare headers based on URL domain
            headers = self.session.headers.copy()
            
            # Domain-specific headers for better compatibility
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
            
            # Fetch image with timeout
            response = self.session.get(
                url,
                headers=headers,
                timeout=Config.IMAGE_FETCH_TIMEOUT,
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
            if self.cache:
                cache_key = LRUImageCache.make_cache_key(url)
                self.cache.put(cache_key, image_data)
            
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
    
    def fetch_from_local(self, path, base_dir=None):
        """
        Fetch image from local file system.
        
        Args:
            path: Local file path (relative or absolute)
            base_dir: Base directory for relative paths
            
        Returns:
            bytes: Image data or None if fetch failed
        """
        try:
            # Security: ensure path is within allowed directories
            if '..' in path:
                print(f"❌ Invalid path (contains ..): {path}")
                return None
            
            # Get base directory
            if base_dir is None:
                base_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Handle relative paths
            if path.startswith('./'):
                path = path[2:]
            
            # Construct absolute path
            full_path = os.path.join(base_dir, path)
            
            # Verify path is within the base directory (security check)
            if not os.path.abspath(full_path).startswith(base_dir):
                print(f"❌ Invalid path (outside base directory): {path}")
                return None
            
            if not os.path.exists(full_path):
                print(f"❌ Image not found: {full_path}")
                return None
            
            # Verify it's actually an image file
            if not full_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
                print(f"❌ Invalid file type: {path}")
                return None
            
            # Read image file
            with open(full_path, 'rb') as img_file:
                return img_file.read()
        
        except IOError as e:
            print(f"❌ Cannot read image {path}: {e}")
            return None
        except Exception as e:
            print(f"❌ Error fetching local image {path}: {e}")
            return None
    
    def to_base64(self, image_data, path_or_url):
        """
        Convert image data to base64 data URI.
        
        Args:
            image_data: Raw image bytes
           path_or_url: Path or URL to detect mime type
            
        Returns:
            str: Base64 data URI
        """
        # Detect mime type from extension
        ext = path_or_url.lower().split('.')[-1]
        mime_type_map = {
            'png': 'image/png',
            'gif': 'image/gif',
            'webp': 'image/webp',
            'bmp': 'image/bmp',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg'
        }
        mime_type = mime_type_map.get(ext, 'image/jpeg')
        
        # Encode to base64
        img_base64 = base64.b64encode(image_data).decode('utf-8')
        return f"data:{mime_type};base64,{img_base64}"
