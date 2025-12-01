import torch
import requests
import os
from PIL import Image
from psycopg.rows import dict_row
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader, Dataset

CACHE = "../.cache"
MODEL_LIST = [  "openai/clip-vit-base-patch32",
                "openai/clip-vit-base-patch16",
                "openai/clip-vit-large-patch14",
                "openai/clip-vit-large-patch14-336" ]

class Image_Embedder:
    def __init__(self, MODEL_ID: int):
        MODEL = MODEL_LIST[MODEL_ID]
        self.processor = CLIPProcessor.from_pretrained(MODEL, use_fast=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(MODEL).to(self.device)
        self.model.eval()

    def convert_from_path(self, image_path: str):
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            image_embeddings = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            final_vector = image_embeddings.cpu().numpy()
            del inputs
            del image_features
            del image_embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return final_vector[0]
        except:
            return None
    
    def convert_from_url(self, url: str):
        response = requests.get(url)

        if response.status_code == 200:
            with open("hnsw_search/user_data/images/temp.jpg", 'wb') as file:
                file.write(response.content)
            print("Đã lưu ảnh thành công!")
        else:
            print("Không tải được ảnh.")
            return None

        vector = self.convert_from_path("hnsw_search/user_data/images/temp.jpg")
        os.remove("hnsw_search/user_data/images/temp.jpg")
        return vector
    
    def convert_from_text(self, text: str):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        vector_result = text_features[0].cpu().numpy()
        return vector_result
    
    