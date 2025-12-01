import hnswlib
import h5py
import os
import requests
import shutil
import numpy as np
from . import embedder

class Search:
    def __init__(self, file_path, dim, num_elements, reset: bool):
        self.user_folder = os.path.dirname(file_path) + "/user_data"
        if reset and os.path.exists(file_path + '.bin'):
            shutil.rmtree(self.user_folder)
            os.remove(file_path + '.bin')
        if not os.path.exists(self.user_folder):
            os.makedirs(self.user_folder)
        self.data = h5py.File(file_path, 'r')
        self.base = len(self.data['embeddings'])
        self.file_path = file_path
        if os.path.exists(file_path + '.bin'):
            self.p = hnswlib.Index(space='ip', dim=dim)
            self.p.load_index(file_path + '.bin')
            self.p.set_ef(200)
        else:
            ids = np.arange(len(self.data['embeddings']))
            self.p = hnswlib.Index(space='ip', dim=dim)
            self.p.init_index(max_elements=num_elements, ef_construction=200, M=20)
            self.p.add_items(self.data['embeddings'], ids)
        self.checkpoint()
    
    def search(self, query, k):
        query_vector = np.array(query, dtype=np.float32)
        idx, distance = self.p.knn_query(query_vector, k=k)
        idx = idx[0]
        distance = distance[0]
        list_images = []
        for i in idx:
            if i < self.base:
                list_images.append(self.data['urls'][i])
            else:
                list_images.append(self.user_folder + f"/{i - self.base}.jpg")
        similarity_scores = [1 - d for d in distance]
        return list_images, similarity_scores
    
    def checkpoint(self):
        self.p.save_index(self.file_path + '.bin')

    def add_url(self, url, model_id):
        response = requests.get(url)
        if response.status_code == 200:
            with open(self.user_folder + f"/{self.p.element_count - self.base}.jpg", 'wb') as file:
                file.write(response.content)
            convert = embedder.Image_Embedder(model_id)
            vector = convert.convert_from_path(self.user_folder + f"/{self.p.element_count - self.base}.jpg")
            self.p.add_items([vector], [self.p.element_count])
            del convert
            self.checkpoint()
            return True
        else:
            return False
