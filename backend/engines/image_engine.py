"""
CLIP-based image search engine.
"""

import torch
import clip
import numpy as np
from PIL import Image

from core.base_engine import BaseSearchEngine
from core.config import Config


class ImageSearchEngine(BaseSearchEngine):
    """
    CLIP ViT-B/32 based image search engine.
    Supports both text queries and image queries.
    """
    
    def __init__(self):
        """Initialize image search engine."""
        super().__init__()
        print("🚀 Initializing CLIP Image Search Engine...")
        self.load_model()
        self.load_index(Config.IMAGE_H5_PATH, item_key='image_path')
        print("✅ Image Search Engine ready!")
    
    def load_model(self):
        """Load CLIP ViT-B/32 model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📱 Device: {self.device}")
        print("📥 Loading CLIP model...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        print("✅ CLIP model loaded")
    
    def text_to_vector(self, text):
        """
        Convert text to CLIP embedding.
        
        Args:
            text: Text query string
            
        Returns:
            np.ndarray: CLIP embedding vector
        """
        text_tokens = clip.tokenize([text.strip()]).to(self.device)
        with torch.no_grad():
            vector_text = self.model.encode_text(text_tokens)
            vector_text = vector_text / vector_text.norm(dim=-1, keepdim=True)
        return vector_text.cpu().numpy().astype(np.float32)
    
    def image_to_vector(self, image):
        """
        Convert PIL image to CLIP embedding.
        
        Args:
            image: PIL Image object
            
        Returns:
            np.ndarray: CLIP embedding vector or None on error
        """
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
        """
        Search for similar images using uploaded image.
        
        Args:
            image: PIL Image object
            k: Number of results to return
            
        Returns:
            list: List of dicts with 'path', 'score' keys
        """
        if self.index is None:
            return []
        
        try:
            # Convert image to vector
            query_vector = self.image_to_vector(image)
            if query_vector is None:
                return []
            
            # Search in HNSW index
            indices, distances = self.index.knn_query(query_vector, k=k)
            
            # Build results
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                item = self.items[idx]
                item_str = item.decode('utf-8') if isinstance(item, bytes) else str(item)
                item_str = item_str.strip()
                
                similarity_score = 1 - distance
                results.append({
                    'path': item_str,
                    'url': item_str,  # For compatibility
                    'score': float(similarity_score)
                })
            
            return results
        
        except Exception as e:
            print(f"❌ Image search error: {e}")
            import traceback
            traceback.print_exc()
            return []
