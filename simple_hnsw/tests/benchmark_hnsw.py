import sys, os
import time
import numpy as np
import argparse

# Add project root to path to import modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(project_root, 'src'))

from simple_hnsw import hnsw
import hnswlib

def calculate_recall(ground_truth_indices, prediction_indices):
    """
    Calculate Recall@K assuming prediction_indices contains lists of neighbors for each query.
    """
    total_recall = 0
    num_queries = len(ground_truth_indices)
    
    for gt, pred in zip(ground_truth_indices, prediction_indices):
        gt_set = set(gt)
        pred_set = set(pred)
        intersection = gt_set.intersection(pred_set)
        if len(gt_set) == 0:
            recall = 0
        else:
            recall = len(intersection) / len(gt_set)
        total_recall += recall
        
    return total_recall / num_queries

def benchmark(max_elements=1000, dim=128, n_queries=10, K=10, M=16, ef_construction=100, ef_search=50, runs=1):
    print(f"Benchmarking with: Elements={max_elements}, Dim={dim}, Queries={n_queries}, K={K}")
    print(f"Params: M={M}, ef_construction={ef_construction}, ef_search={ef_search}")
    
    # Generate random data
    train_data = np.random.rand(max_elements, dim).astype(np.float32)
    query_data = np.random.rand(n_queries, dim).astype(np.float32)
    
    # --- Run hnswlib (Reference) ---
    print("\n--- Running hnswlib (Reference) ---")
    start_time = time.time()
    lib_index = hnswlib.Index('l2', dim)
    lib_index.init_index(max_elements=max_elements, M=M, ef_construction=ef_construction)
    lib_index.add_items(train_data)
    lib_build_time = time.time() - start_time
    print(f"hnswlib Build Time: {lib_build_time:.4f}s")
    
    lib_index.set_ef(ef_search)
    
    start_time = time.time()
    # hnswlib returns (indices, distances)
    lib_labels, _ = lib_index.knn_query(query_data, k=K) 
    lib_search_time = time.time() - start_time
    
    print(f"hnswlib Search Time: {lib_search_time:.4f}s")
    print(f"hnswlib Avg Query Time: {lib_search_time/n_queries:.6f}s")
    
    
    # --- Run Simple HNSW ---
    print("\n--- Running Simple HNSW (Python Implementation) ---")
    start_time = time.time()
    simple_index = hnsw.HNSW('l2', dim)
    simple_index.init_index(max_elements=max_elements, M=M, ef_construction=ef_construction)
    simple_index.insert_items(train_data)
    simple_build_time = time.time() - start_time
    print(f"Simple HNSW Build Time: {simple_build_time:.4f}s")
    
    # Note: simple_hnsw.hnsw.HNSW usually uses internal settings for search, 
    # but let's check if we can set ef. The class has self.ef initialized in init_index.
    simple_index.ef = ef_search
    
    simple_labels = []
    start_time = time.time()
    for q in query_data:
        # returns list of indices
        lbls = simple_index.knn_search(q, K=K)
        simple_labels.append(lbls)
    simple_search_time = time.time() - start_time
    
    print(f"Simple HNSW Search Time: {simple_search_time:.4f}s")
    print(f"Simple HNSW Avg Query Time: {simple_search_time/n_queries:.6f}s")
    
    # --- Comparison ---
    print("\n--- Comparison ---")
    speedup_build = lib_build_time / simple_build_time if simple_build_time > 0 else 0
    speedup_search = lib_search_time / simple_search_time if simple_search_time > 0 else 0
    print(f"Build Time Ratio (hnswlib / Simple): {speedup_build:.4f} (Lower is better for Simple)")
    print(f"Search Time Ratio (hnswlib / Simple): {speedup_search:.4f}")
    
    # Calculate Recall using hnswlib results as Ground Truth
    # (Note: hnswlib is approximate, but usually very high quality. 
    # Ideally compare against Brute Force, but this script compares two HNSW implementations)
    recall = calculate_recall(lib_labels, simple_labels)
    print(f"Recall@K (relative to hnswlib): {recall:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark Simple HNSW vs hnswlib')
    parser.add_argument('--N', type=int, default=1000, help='Number of elements')
    parser.add_argument('--dim', type=int, default=64, help='Dimension of vectors')
    parser.add_argument('--queries', type=int, default=10, help='Number of queries')
    parser.add_argument('--K', type=int, default=10, help='K nearest neighbors')
    
    args = parser.parse_args()
    
    benchmark(max_elements=args.N, dim=args.dim, n_queries=args.queries, K=args.K)