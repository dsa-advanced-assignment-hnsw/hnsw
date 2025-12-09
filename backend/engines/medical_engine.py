"""
BiomedCLIP-based medical image search engine.
"""

import torch
import open_clip
import numpy as np
from PIL import Image

from core.base_engine import BaseSearchEngine
from core.config import Config


class MedicalSearchEngine(BaseSearchEngine):
    """
    BiomedCLIP-based medical image search engine for bone fracture detection.
    """
    
    def __init__(self):
        """Initialize medical search engine."""
        super().__init__()
        print("🚀 Initializing BiomedCLIP Medical Search Engine...")
        self.load_model()
        self.load_index(Config.MEDICAL_H5_PATH, item_key='image_path')
        print("✅ Medical Search Engine ready!")
    
    def load_model(self):
        """Load BiomedCLIP model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📱 Device: {self.device}")
        print("📥 Loading BiomedCLIP model...")
        
        # Load BiomedCLIP model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = open_clip.get_tokenizer(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        
        print("✅ BiomedCLIP model loaded")
    
    def text_to_vector(self, text):
        """
        Convert medical text to BiomedCLIP embedding.
        
        Args:
            text: Medical text query
            
        Returns:
            np.ndarray: BiomedCLIP embedding vector
        """
        text_tokens = self.tokenizer([text.strip()]).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy().astype(np.float32)
    
    def image_to_vector(self, image):
        """
        Convert PIL image to BiomedCLIP embedding.
        
        Args:
            image: PIL Image object
            
        Returns:
            np.ndarray: BiomedCLIP embedding vector or None on error
        """
        try:
            # Preprocess image
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"❌ Image encoding error: {e}")
            return None
    
    def search_by_image(self, image, k=20):
        """
        Search for similar medical images using uploaded X-ray.
        
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
                    'score': float(similarity_score)
                })
            
            return results
        
        except Exception as e:
            print(f"❌ Medical image search error: {e}")
            import traceback
            traceback.print_exc()
            return []
