import h5py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from . import text
from . import image

class brute_force_search:
    def __init__(self):
        self.image_paths = None
        self.image_vectors = None

        self.text_paths = None
        self.text_vectors = None

    def read_image_converted_vectors(self, file_path):
        with h5py.File(file_path, "r") as f:
            self.image_paths = np.array([p.decode("utf-8") for p in f["path"][:]])
            self.image_vectors = np.array(f["embeddings"][:])

    def read_text_converted_vectors(self, file_path):
        with h5py.File(file_path, "r") as f:
            self.text_paths = np.array([p.decode("utf-8") for p in f["path"][:]])
            self.text_vectors = np.array(f["embeddings"][:])

    def search_image(self, qtext):
        text_vector = image.embed_Text(qtext)
        sims = cosine_similarity(text_vector.reshape(1, -1), self.image_vectors)[0]
        best_idx = np.argmax(sims)
        return self.image_paths[best_idx], sims[best_idx]

    def search_topK_images(self, qtext, k=5):
        text_vector = image.embed_Text(qtext)
        sims = cosine_similarity(text_vector.reshape(1, -1), self.image_vectors)[0]
        topk_idx = np.argsort(sims)[::-1][:k]
        return [(self.image_paths[i], sims[i]) for i in topk_idx]

    def search_text(self, qtext):
        text_vector = text.embed_Text(qtext)
        sims = cosine_similarity(text_vector.reshape(1, -1), self.text_vectors)[0]
        best_idx = np.argmax(sims)
        return self.text_paths[best_idx], sims[best_idx]

    def search_topK_texts(self, qtext, k=5):
        text_vector = text.embed_Text(qtext)
        sims = cosine_similarity(text_vector.reshape(1, -1), self.text_vectors)[0]
        topk_idx = np.argsort(sims)[::-1][:k]
        return [(self.text_paths[i], sims[i]) for i in topk_idx]
