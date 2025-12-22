import numpy as np
import pytest
from brute_force.brute_force_search import brute_force_search

def test_brute_force_search_l2():
    # Create simple data
    # (0,0), (1,1), (2,2)
    train_data = np.array([[0,0], [1,1], [2,2]])
    query_vec = np.array([[0.1, 0.1]])
    
    # 0,0 should be closest
    indices, distances = brute_force_search('l2', train_data, query_vec, K=1)
    
    assert indices[0][0] == 0

def test_brute_force_search_cosine():
    train_data = np.array([[1,0], [0,1], [-1,0]])
    query_vec = np.array([[1, 0.1]]) # Very close to (1,0)
    
    indices, distances = brute_force_search('cosine', train_data, query_vec, K=1)
    
    assert indices[0][0] == 0

def test_brute_force_multiple_queries():
    train_data = np.random.rand(10, 5)
    query_vecs = np.random.rand(3, 5)
    
    k = 2
    indices, distances = brute_force_search('l2', train_data, query_vecs, K=k)
    
    assert indices.shape == (3, k)
    assert distances.shape == (3, k)

def test_brute_force_k_larger_than_data():
    train_data = np.random.rand(5, 5)
    query_vec = np.random.rand(1, 5)
    
    # K=10 but only 5 items
    indices, distances = brute_force_search('l2', train_data, query_vec, K=10)
    
    # It seems the implementation slices [:K], so it should return min(N, K)
    assert indices.shape == (1, 5)
