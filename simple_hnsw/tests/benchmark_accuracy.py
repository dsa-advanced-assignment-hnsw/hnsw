import sys, os
import time
import numpy as np

# Add project root to path to import modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(project_root, 'src'))

from simple_hnsw.hnsw import HNSW
from brute_force.brute_force_search import brute_force_search

def calculate_recall(ground_truth_indices, prediction_indices):
    """
    Calculate Recall@K for a set of queries.
    Recall = (Intersection of Truth and Prediction) / (Size of Truth)
    Here we assume K is the same for both.
    """
    total_recall = 0
    num_queries = len(ground_truth_indices)
    
    for gt, pred in zip(ground_truth_indices, prediction_indices):
        gt_set = set(gt)
        pred_set = set(pred)
        intersection = gt_set.intersection(pred_set)
        recall = len(intersection) / len(gt_set)
        total_recall += recall
        
    return total_recall / num_queries

def benchmark_recall_time(dim=128, max_elements=1000, n_queries=10, K=10, M=16, ef_construction=100, ef_search=50):
    print(f"Benchmarking with: Elements={max_elements}, Dim={dim}, Queries={n_queries}, K={K}")
    print(f"HNSW Params: M={M}, ef_construction={ef_construction}, ef_search={ef_search}")
    
    # Generate random data
    train_data = np.random.rand(max_elements, dim).astype(np.float32)
    query_data = np.random.rand(n_queries, dim).astype(np.float32)
    
    # --- Brute Force (Ground Truth) ---
    print("\n--- Running Brute Force Search ---")
    start_time = time.time()
    bf_indices, bf_distances = brute_force_search('l2', train_data, query_data, K=K)
    bf_time = time.time() - start_time
    print(f"Brute Force Time: {bf_time:.4f}s")
    print(f"Avg Brute Force Query Time: {bf_time/n_queries:.6f}s")
    
    # --- HNSW ---
    print("\n--- Running HNSW ---")
    hnsw = HNSW('l2', dim=dim)
    
    build_start_time = time.time()
    hnsw.init_index(max_elements=max_elements, M=M, ef_construction=ef_construction)
    hnsw.insert_items(train_data)
    build_time = time.time() - build_start_time
    print(f"HNSW Build Time: {build_time:.4f}s")
    
    # HNSW Search
    # Note: simple_hnsw.hnsw.HNSW class has 'ef' attribute, defaulting to 10. 
    # We should set it higher for better recall if needed, or pass it if search implementation allows.
    # Looking at the code, it uses self.ef or can be passed? 
    # search_layer takes 'ef', but knn_search uses self.ef for the final search layer (0).
    hnsw.ef = ef_search 
    
    hnsw_indices = []
    
    search_start_time = time.time()
    for query in query_data:
        # knn_search returns just a list of indices
        neighbors = hnsw.knn_search(query, K=K)
        hnsw_indices.append(neighbors)
    hnsw_search_time = time.time() - search_start_time
    
    print(f"HNSW Search Time: {hnsw_search_time:.4f}s")
    print(f"Avg HNSW Query Time: {hnsw_search_time/n_queries:.6f}s")
    print(f"Speedup vs Brute Force: {bf_time / hnsw_search_time:.2f}x")
    
    # --- Calculate Recall ---
    recall = calculate_recall(bf_indices, hnsw_indices)
    print(f"\nRecall@{K}: {recall:.4f}")
    
    return recall

if __name__ == "__main__":
    # Small benchmark
    benchmark_recall_time(dim=64, max_elements=10000, n_queries=20, K=100)
