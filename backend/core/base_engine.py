"""
Abstract base class for search engines.
Provides common functionality for HNSW-based semantic search.
"""

from abc import ABC, abstractmethod
import numpy as np
import hnswlib
import h5py
import os

from core.config import Config


class BaseSearchEngine(ABC):
    """
    Abstract base class for HNSW-based search engines.
    Subclasses must implement model loading and encoding methods.
    """
    
    def __init__(self):
        """Initialize base search engine."""
        self.index = None
        self.items = None  # URLs, paths, or other item identifiers
        self.device = None
        self.model = None
    
    @abstractmethod
    def load_model(self):
        """
        Load the ML model (CLIP, RoBERTa, BiomedCLIP, etc.).
        Must set self.model and self.device.
        """
        pass
    
    @abstractmethod
    def text_to_vector(self, text):
        """
        Convert text query to embedding vector.
        
        Args:
            text: Text query string
            
        Returns:
            np.ndarray: Embedding vector (float32)
        """
        pass
    
    def load_index(self, h5_path, item_key='image_path'):
        """
        Load HNSW index from H5 file or .bin file if available.
        
        Args:
            h5_path: Path to H5 file containing embeddings
            item_key: Key in H5 file for item paths/URLs (default: 'image_path')
        """
        try:
            # Determine the .bin filename
            index_filename = os.path.splitext(h5_path)[0] + ".bin"
            
            # Check if .h5 file exists
            if not os.path.exists(h5_path):
                print(f"❌ Error: {h5_path} not found!")
                return
            
            print(f"📂 Loading embeddings from: {h5_path}")
            
            # Load embeddings and items
            with h5py.File(h5_path, "r") as f:
                print(f"   Available datasets: {list(f.keys())}")
                
                # Try to load items (supports multiple key names)
                item_keys_to_try = [item_key, 'urls', 'image_url', 'image_path', 'paths']
                for key in item_keys_to_try:
                    if key in f:
                        self.items = f[key][:]
                        print(f"   → Loaded {len(self.items)} items from '{key}'")
                        break
                
                if self.items is None:
                    raise ValueError(f"No item dataset found. Tried: {item_keys_to_try}")
                
                embs = f["embeddings"][:]
                print(f"   → Loaded embeddings shape: {embs.shape}")
            
            # Sample items to verify format
            sample_items = self.items[:3]
            print("   Sample items:")
            for i, item in enumerate(sample_items):
                decoded = item.decode('utf-8') if isinstance(item, bytes) else str(item)
                print(f"      {i+1}. {decoded[:80]}...")
            
            dim = embs.shape[1]
            
            # Check if .bin file exists for faster loading
            if os.path.exists(index_filename):
                print(f"⚡ Loading HNSW index from: {index_filename}")
                self.index = hnswlib.Index(space='cosine', dim=dim)
                self.index.load_index(index_filename, max_elements=Config.MAX_HNSW_ELEMENTS)
                self.index.set_ef(Config.HNSW_EF_SEARCH)
                print(f"✅ HNSW index loaded from .bin file!")
            else:
                # Build index from scratch
                print(f"🔧 Building HNSW index from scratch...")
                print(f"   M={Config.HNSW_M}, ef_construction={Config.HNSW_EF_CONSTRUCTION}, ef={Config.HNSW_EF_SEARCH}")
                self.index = hnswlib.Index(space='cosine', dim=dim)
                self.index.init_index(
                    max_elements=Config.MAX_HNSW_ELEMENTS,
                    ef_construction=Config.HNSW_EF_CONSTRUCTION,
                    M=Config.HNSW_M
                )
                self.index.set_ef(Config.HNSW_EF_SEARCH)
                self.index.add_items(embs)
                # Save index for future runs
                self.index.save_index(index_filename)
                print(f"💾 Saved HNSW index to: {index_filename}")
            
            print(f"✅ Search engine ready! Indexed {len(self.items):,} items")
            print(f"📊 Dimension: {dim}, Device: {self.device}")
        
        except Exception as e:
            print(f"❌ Error loading index: {e}")
            import traceback
            traceback.print_exc()
    
    def search(self, query, k=20):
        """
        Search for similar items using text query.
        
        Args:
            query: Text query string
            k: Number of results to return
            
        Returns:
            list: List of dicts with 'path', 'score' keys
        """
        if self.index is None:
            return []
        
        try:
            # Convert query to vector
            query_vector = self.text_to_vector(query)
            
            # Search in HNSW index
            indices, distances = self.index.knn_query(query_vector, k=k)
            
            # Build results
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                item = self.items[idx]
                item_str = item.decode('utf-8') if isinstance(item, bytes) else str(item)
                item_str = item_str.strip()
                
                similarity_score = 1 - distance  # Convert distance to similarity
                results.append({
                    'path': item_str,
                    'score': float(similarity_score)
                })
            
            return results
        
        except Exception as e:
            print(f"❌ Search error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_stats(self):
        """
        Get search engine statistics.
        
        Returns:
            dict: Engine statistics
        """
        return {
            'model_loaded': self.model is not None,
            'index_loaded': self.index is not None,
            'device': str(self.device) if self.device else None,
            'total_items': len(self.items) if self.items is not None else 0,
            'hnsw_config': {
                'M': Config.HNSW_M,
                'ef_construction': Config.HNSW_EF_CONSTRUCTION,
                'ef_search': Config.HNSW_EF_SEARCH,
                'max_elements': Config.MAX_HNSW_ELEMENTS
            }
        }
