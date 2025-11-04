import torch
import clip
import h5py
import os
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

class EmbedX:
    def __init__(self):
        # Automatically select device: use GPU (CUDA) if available, otherwise CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load CLIP model (ViT-B/32) and preprocessing pipeline, placed on the chosen device
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        # Paths and storage for image dataset and image embeddings
        self.dataset_path = ""
        self.embedding_vector_path = ""

    def select_Dataset_Path(self, dataset_path):
        self.dataset_path = dataset_path

    def select_Output_Path(self, embedding_vector_path):
        self.embedding_vector_path = embedding_vector_path
        if not os.path.exists(embedding_vector_path):
            with h5py.File(self.embedding_vector_path, "w") as outfile:
                outfile.create_dataset(
                    "path",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=h5py.string_dtype(encoding="utf-8")
                )
                
                outfile.create_dataset(
                    "embeddings",
                    shape=(0, 512),
                    maxshape=(None, 512),
                    dtype=np.float32,
                    chunks=True
                )

    def embed_Dataset(self):
        total = sum(1 for file in os.listdir(self.dataset_path) if os.path.isfile(os.path.join(self.dataset_path,file)))
        pbar = tqdm(total = total, desc = "Embedding: ")
        for filename in os.listdir(self.dataset_path):
            file_path = self.dataset_path + "/" + filename
            image = self.preprocess(Image.open(file_path)).unsqueeze(0).to(self.device)
            vector = self.model.encode_image(image)
            vector = vector / vector.norm(dim=-1, keepdim=True)
            vector = vector.cpu().detach().numpy().astype(np.float32)

            with h5py.File(self.embedding_vector_path, "a") as f:
                idx = f["path"].shape[0]

                f["path"].resize(idx + 1, axis=0)
                f["embeddings"].resize(idx + 1, axis=0)

                f["path"][idx] = file_path
                f["embeddings"][idx] = vector
            pbar.update(1)
        pbar.close()

    def embed_Other_file(self, file_path):
        image = self.preprocess(Image.open(file_path)).unsqueeze(0).to(self.device)
        vector = self.model.encode_image(image)
        vector = vector / vector.norm(dim=-1, keepdim=True)
        vector = vector.cpu().detach().numpy().astype(np.float32)

        with h5py.File(self.embedding_vector_path, "a") as f:
            idx = f["path"].shape[0]

            f["path"].resize(idx + 1, axis=0)
            f["embeddings"].resize(idx + 1, axis=0)

            f["path"][idx] = file_path
            f["embeddings"][idx] = vector

def embed_Image(file_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)
    vector = model.encode_image(image)
    vector = vector / vector.norm(dim=-1, keepdim=True)
    vector = vector.cpu().detach().numpy().astype(np.float32)
    return vector

def embed_Text(text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    text_tokens = clip.tokenize([text]).to(device)
    vector = model.encode_text(text_tokens)
    vector = vector / vector.norm(dim=-1, keepdim=True)
    vector = vector.cpu().detach().numpy().astype(np.float32)
    return vector[0]