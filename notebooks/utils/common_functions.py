"""
Common utility functions shared across notebooks.
Extract reusable code from notebooks here.
"""

import numpy as np
import h5py


def load_embeddings_from_h5(filepath, embedding_key='embeddings', path_key='image_path'):
    """
    Load embeddings and paths from H5 file.
    
    Args:
        filepath: Path to H5 file
        embedding_key: Key for embeddings dataset
        path_key: Key for paths dataset
        
    Returns:
        tuple: (embeddings array, paths array)
    """
    with h5py.File(filepath, 'r') as f:
        embeddings = f[embedding_key][:]
        paths = f[path_key][:]
    
    return embeddings, paths


def save_embeddings_to_h5(filepath, embeddings, paths, embedding_key='embeddings', path_key='image_path'):
    """
    Save embeddings and paths to H5 file.
    
    Args:
        filepath: Path to output H5 file
        embeddings: Numpy array of embeddings
        paths: Array of file paths or URLs
        embedding_key: Key for embeddings dataset
        path_key: Key for paths dataset
    """
    with h5py.File(filepath, 'w') as f:
        f.create_dataset(embedding_key, data=embeddings, compression='gzip')
        if isinstance(paths[0], str):
            paths = [p.encode('utf-8') for p in paths]
        f.create_dataset(path_key, data=paths)
    
    print(f"✅ Saved {len(embeddings)} embeddings to {filepath}")


def normalize_embeddings(embeddings):
    """
    L2 normalize embeddings.
    
    Args:
        embeddings: Numpy array of embeddings
        
    Returns:
        Normalized embeddings
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


def compute_similarity(query_embedding, embeddings):
    """
    Compute cosine similarity between query and all embeddings.
    
    Args:
        query_embedding: Single embedding vector
        embeddings: Array of embeddings
        
    Returns:
        Array of similarity scores
    """
    # Normalize
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    embeddings_norm = normalize_embeddings(embeddings)
    
    # Compute cosine similarity
    similarities = np.dot(embeddings_norm, query_norm)
    
    return similarities
