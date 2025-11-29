import hnswlib
import h5py
import numpy as np

dim = 512     
num_elements = 20000 

class Search:
    def __init__(self, file_path):
        self.data = h5py.File(file_path, 'r')
        ids = np.arange(len(self.data['embeddings']))
        self.p = hnswlib.Index(space='ip', dim=dim)
        self.p.init_index(max_elements=num_elements, ef_construction=200, M=20)
        self.p.add_items(self.data['embeddings'], ids)
        self.max_id = max(ids)
    
    def search(self, query, k):
        query_vector = np.array(query, dtype=np.float32)
        idx, distance = self.p.knn_query(query_vector, k=k)
        idx = idx[0]
        distance = distance[0]
        list_images = [self.data['urls'][i] for i in idx]
        similarity_scores = [1 - d for d in distance]
        return list_images, similarity_scores