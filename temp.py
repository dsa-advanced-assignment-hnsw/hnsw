import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Dict, Any, List

# --- CẤU HÌNH ---
file_configs = {
    # Nhóm 512 chiều
    'clip-vit-base-patch32': {'filename': '.cache/clip-vit-base-patch32/image_embeddings/clip-vit-base-patch32_Images_Embedded_1_to_100000.h5', 'dim': 512, 'group': '512D'},
    'clip-vit-base-patch16': {'filename': '.cache/clip-vit-base-patch16/image_embeddings/clip-vit-base-patch16_Images_Embedded_1_to_100000.h5', 'dim': 512, 'group': '512D'},
    # Nhóm 768 chiều
    'clip-vit-large-patch14': {'filename': '.cache/clip-vit-large-patch14/image_embeddings/clip-vit-large-patch14_Images_Embedded_1_to_100000.h5', 'dim': 768, 'group': '768D'},
    'clip-vit-large-patch14-336': {'filename': '.cache/clip-vit-large-patch14-336/image_embeddings/clip-vit-large-patch14-336_Images_Embedded_1_to_10000.h5', 'dim': 768, 'group': '768D'},
}

EMBEDDING_KEY = 'embeddings' # Tên trường chứa vector nhúng trong tệp H5
SAMPLE_SIZE = 10000 
NUM_CLUSTERS_KMEANS = 4 # Số cụm giả định để đánh giá phân biệt

def load_and_sample_embeddings(filename: str, model_name: str, expected_dim: int, sample_size: int) -> np.ndarray | None:
    """Tải và lấy K phần tử đầu tiên (first K sampling) của các embeddings từ tệp H5."""
    try:
        # --- LOGIC TẢI TỪ H5PY ---
        with h5py.File(filename, 'r') as f:
            if EMBEDDING_KEY not in f:
                 raise KeyError(f"Field '{EMBEDDING_KEY}' does not exist in the file.")
            embeddings = f[EMBEDDING_KEY][:]
        # ------------------------------------
        
        current_count = embeddings.shape[0]

        if embeddings.shape[1] != expected_dim:
            print(f"Size error: {model_name} has dimension {embeddings.shape[1]}, which does not match {expected_dim}. Skipping.")
            return None

        # --- ĐIỀU CHỈNH: LẤY MẪU ĐẦU TIÊN (FIRST K) ---
        if current_count >= sample_size:
            print(f"[{model_name}]: Taking the first {sample_size} points.")
            embeddings = embeddings[:sample_size]
        elif current_count < sample_size:
             # Nếu số lượng ít hơn SAMPLE_SIZE, lấy tất cả
             print(f"[{model_name}]: Taking all {current_count} points (less than {sample_size}).")
        # ---------------------------------------------

        # Normalization (Crucial if your vectors are not already unit length, e.g., for Cosine Similarity)
        norm_check = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norm_check, 1.0, atol=1e-3):
             embeddings = embeddings / norm_check[:, np.newaxis]

        return embeddings
    except FileNotFoundError:
        print(f"ERROR: File {filename} not found. Please check the file name.")
        return None
    except KeyError as e:
        print(f"ERROR: {e} in file {filename}.")
        return None
    except Exception as e:
         print(f"OTHER ERROR loading {filename}: {e}")
         return None

# --- LOAD AND GROUP DATA ---
data_by_group: Dict[str, List[pd.DataFrame]] = {'512D': [], '768D': []}
model_embeddings: Dict[str, np.ndarray] = {}
stats: Dict[str, Dict[str, Any]] = {}

for model_name, config in file_configs.items():
    embeddings = load_and_sample_embeddings(
        config['filename'],
        model_name,
        config['dim'],
        SAMPLE_SIZE
    )
    if embeddings is not None:
        model_embeddings[model_name] = embeddings
        
        data_by_group[config['group']].append(
            pd.DataFrame({
                'embedding': list(embeddings),
                'model': model_name
            })
        )

        stats[model_name] = {
            'Count': embeddings.shape[0],
            'Embedding_Dim': embeddings.shape[1],
            'Mean_Norm': np.mean(np.linalg.norm(embeddings, axis=1)), # Average vector magnitude
            'Mean': np.mean(embeddings),
            'StdDev': np.std(embeddings),
        }