import fitz  # Library for handling PDF files (PyMuPDF)
import docx  # Library for handling DOCX files (python-docx)
import torch
import numpy as np
import os
import re 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

# --- 1. UNIFIED DOCUMENT READING FUNCTION ---

def read_document_text(file_path):
    """Reads the full text content from .pdf, .docx, and .txt formats."""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    full_text = ""

    if ext == '.pdf':
        try:
            with fitz.open(file_path) as doc:
                for page in doc:
                    full_text += page.get_text()
        except Exception as e:
            print(f"❌ Error reading PDF: {e}")
            return None
            
    elif ext == '.docx':
        try:
            doc = docx.Document(file_path)
            # Iterate through paragraphs to extract all text
            for para in doc.paragraphs:
                full_text += para.text + '\n'
        except Exception as e:
            print(f"❌ Error reading DOCX: {e}")
            return None
            
    elif ext == '.txt':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                full_text = f.read()
        except Exception as e:
            print(f"❌ Error reading TXT: {e}")
            return None
            
    else:
        print(f"❌ Unsupported file format: {ext}")
        return None
        
    # Clean the text: replace multiple spaces/newlines with a single space
    return " ".join(full_text.split())

# --- 2. MAIN CLASS: PaperVectorProcessor ---

class PaperVectorProcessor:
    """
    Processes documents (PDF/DOCX/TXT) by summarizing the entire content 
    and converting that summary into a normalized embedding vector.
    """
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Running on device: {self.device}. Starting model load (one-time)...")

        # Configuration for the LED Base Summarizer (Lower VRAM requirement)
        sum_model_name = "allenai/led-base-16384"
        
        # --- LOAD MODEL 1: SUMMARIZER ---
        self.sum_tokenizer = AutoTokenizer.from_pretrained(sum_model_name)
        
        if self.device == "cuda":
            print(f"  -> Loading Summarizer ({sum_model_name}) with FP16 for VRAM efficiency...")
            self.sum_model = AutoModelForSeq2SeqLM.from_pretrained(
                sum_model_name,
                # Use FP16 to halve VRAM usage
                torch_dtype=torch.float16, 
                low_cpu_mem_usage=True
            ).to(self.device)
        else:
            print(f"  -> Loading Summarizer ({sum_model_name}) on CPU (will be slow)...")
            self.sum_model = AutoModelForSeq2SeqLM.from_pretrained(sum_model_name).to(self.device)

        # --- LOAD MODEL 2: EMBEDDER ---
        print("  -> Loading Embedder (all-roberta-large-v1)...")
        self.embed_model = SentenceTransformer('all-roberta-large-v1', device=self.device)
        print("✅ Models loaded successfully.")

    def _summarize_long_text(self, text):
        """Internal function: Uses LED to summarize the long document text."""
        
        # Tokenize input, truncating if necessary (max 16384 tokens)
        inputs = self.sum_tokenizer(
            text, 
            padding="max_length", 
            truncation=True, 
            max_length=16384, 
            return_tensors="pt"
        )
        
        # Move inputs to the processing device (CPU/GPU)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        # Create Global Attention Mask (required for LED to focus on the start token)
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, 0] = 1 

        # Generate the summary (max length of 450 tokens ensures it fits the 512-token limit of the RoBERTa embedder)
        summary_ids = self.sum_model.generate(
            input_ids, 
            attention_mask=attention_mask, 
            global_attention_mask=global_attention_mask, 
            min_length=100,
            max_length=450, 
            num_beams=4, 
            early_stopping=True
        )

        # Decode the generated tokens back into a string
        summary = self.sum_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def get_vector(self, file_path):
        """
        Public method: Takes a filename (PDF/DOCX/TXT) and returns the 1024-dimensional embedding vector.
        """
        print(f"\n🚀 Processing file: {file_path}")
        
        # 1. Read document text
        full_text = read_document_text(file_path)
        
        if not full_text or len(full_text) < 100: 
            # Skip if the text is empty or too short to be meaningful
            return None

        # 2. Summarize content
        print("2. Summarizing content...")
        generated_abstract = self._summarize_long_text(full_text)
        
        # 3. Embed and L2 Normalize
        print("3. Converting to Vector and Normalizing (L2)...")
        # Generate the embedding vector
        vector = self.embed_model.encode(generated_abstract, convert_to_numpy=True)
        
        # Calculate the L2 norm
        norm = np.linalg.norm(vector)
        
        # Apply L2 Normalization (crucial for consistent cosine similarity search)
        if norm > 0:
            vector = vector / norm
            
        print(f"✅ Successfully processed {os.path.basename(file_path)}. Shape: {vector.shape}")
        return vector