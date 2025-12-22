import numpy as np
import pytest
from simple_hnsw.hnsw import HNSW

@pytest.fixture
def sample_data():
    return np.random.rand(100, 10)  # 100 vectors of dimension 10

def test_hnsw_initialization():
    dim = 10
    hnsw = HNSW('l2', dim)
    hnsw.init_index(max_elements=1000, M=16, ef_construction=100)
    
    assert hnsw.dim == dim
    assert hnsw.max_elements == 1000
    assert hnsw.M == 16
    assert hnsw.ef_construction == 100
    assert len(hnsw.graph) == 0
    assert len(hnsw.data) == 0

def test_insert_single_item():
    dim = 5
    hnsw = HNSW('l2', dim)
    hnsw.init_index(max_elements=10, M=4, ef_construction=50)
    
    vec = np.random.rand(dim)
    hnsw.insert(vec)
    
    assert len(hnsw.data) == 1
    assert np.allclose(hnsw.data[0], vec)
    assert hnsw.cur_element_count == 1
    assert hnsw.entry_point != -1

def test_insert_multiple_items(sample_data):
    dim = 10
    hnsw = HNSW('l2', dim)
    hnsw.init_index(max_elements=200, M=16, ef_construction=100)
    
    hnsw.insert_items(sample_data)
    
    assert len(hnsw.data) == 100
    assert hnsw.cur_element_count == 100

def test_knn_search_l2(sample_data):
    dim = 10
    hnsw = HNSW('l2', dim)
    hnsw.init_index(max_elements=200, M=16, ef_construction=100)
    hnsw.insert_items(sample_data)
    
    # Pick a vector from the data as query
    query_idx = 0
    query_vec = sample_data[query_idx]
    
    # Search for it (k=1)
    neighbors = hnsw.knn_search(query_vec, K=1)
    
    assert len(neighbors) == 1
    # The nearest neighbor to itself should be itself
    assert neighbors[0] == query_idx

def test_knn_search_cosine(sample_data):
    dim = 10
    hnsw = HNSW('cosine', dim)
    hnsw.init_index(max_elements=200, M=16, ef_construction=100)
    hnsw.insert_items(sample_data)
    
    query_idx = 5
    query_vec = sample_data[query_idx]
    
    neighbors = hnsw.knn_search(query_vec, K=1)
    
    assert len(neighbors) == 1
    assert neighbors[0] == query_idx

def test_search_results_structure(sample_data):
    dim = 10
    hnsw = HNSW('l2', dim)
    hnsw.init_index(max_elements=200, M=16, ef_construction=100)
    hnsw.insert_items(sample_data)
    
    query_vec = np.random.rand(dim)
    k = 5
    neighbors = hnsw.knn_search(query_vec, K=k)
    
    assert isinstance(neighbors, list)
    assert len(neighbors) <= k
    # Should perform better than random, but hard to assert strictly in unit test without specific data
    # We mainly check that it returns indices within range
    assert all(0 <= idx < 100 for idx in neighbors)

def test_knn_search_k_larger_than_data():
    dim = 5
    hnsw = HNSW('l2', dim)
    hnsw.init_index(max_elements=100, M=4, ef_construction=50)
    
    data = np.random.rand(5, dim)
    hnsw.insert_items(data)
    
    query_vec = np.random.rand(dim)
    # Ask for more neighbors than exist
    neighbors = hnsw.knn_search(query_vec, K=10)
    
    assert len(neighbors) <= 5
