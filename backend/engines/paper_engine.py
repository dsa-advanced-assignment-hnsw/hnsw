"""
RoBERTa-based paper search engine.
"""

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

from core.base_engine import BaseSearchEngine
from core.config import Config


class PaperSearchEngine(BaseSearchEngine):
    """
    RoBERTa-large based academic paper search engine.
    """
    
    def __init__(self):
        """Initialize paper search engine."""
        super().__init__()
        print("🚀 Initializing RoBERTa Paper Search Engine...")
        self.load_model()
        self.load_index(Config.PAPER_H5_PATH, item_key='urls')
        print("✅ Paper Search Engine ready!")
    
    def load_model(self):
        """Load RoBERTa-large model from Hugging Face."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📱 Device: {self.device}")
        print("📥 Loading RoBERTa model...")
        
        model_name = "sentence-transformers/all-roberta-large-v1"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        print("✅ RoBERTa model loaded")
    
    def text_to_vector(self, text):
        """
        Convert text to RoBERTa embedding.
        
        Args:
            text: Text query string
            
        Returns:
            np.ndarray: RoBERTa embedding vector
        """
        # Tokenize and encode
        inputs = self.tokenizer(
            text.strip(),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            # Normalize
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        return embeddings.cpu().numpy().astype(np.float32)
