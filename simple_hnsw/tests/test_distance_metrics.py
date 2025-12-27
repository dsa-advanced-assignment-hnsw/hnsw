import numpy as np
import pytest
from simple_hnsw.distance_metrics import l2_distance, cosine_distance

def test_l2_distance_basic():
    a = [1, 2, 3]
    b = [4, 5, 6]
    # sqrt((1-4)^2 + (2-5)^2 + (3-6)^2) = sqrt(9 + 9 + 9) = sqrt(27) ≈ 5.196
    expected = np.sqrt(27)
    assert np.isclose(l2_distance(a, b), expected)

def test_l2_distance_numpy_arrays():
    a = np.array([1, 0])
    b = np.array([0, 1])
    # sqrt(1^2 + (-1)^2) = sqrt(2)
    assert np.isclose(l2_distance(a, b), np.sqrt(2))

def test_l2_distance_zero():
    a = [0, 0, 0]
    b = [0, 0, 0]
    assert l2_distance(a, b) == 0.0

def test_cosine_distance_orthogonal():
    a = [1, 0]
    b = [0, 1]
    # Cosine similarity = 0, distance = 1 - 0 = 1
    assert np.isclose(cosine_distance(a, b), 1.0)

def test_cosine_distance_identical():
    a = [1, 2, 3]
    b = [1, 2, 3]
    # Cosine similarity = 1, distance = 1 - 1 = 0
    assert np.isclose(cosine_distance(a, b), 0.0)

def test_cosine_distance_opposite():
    a = [1, 1]
    b = [-1, -1]
    # Cosine similarity = -1, distance = 1 - (-1) = 2
    assert np.isclose(cosine_distance(a, b), 2.0)

def test_cosine_distance_zero_vector():
    a = [0, 0]
    b = [1, 1]
    # Should handle zero vector division by returning 1.0 (max distance)
    assert cosine_distance(a, b) == 1.0
