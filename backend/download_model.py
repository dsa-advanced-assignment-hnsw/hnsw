#!/usr/bin/env python3
"""
Script to download the CLIP model when internet is available.
Run this once when you have internet connection.
"""

import clip
import torch

print("🔄 Downloading CLIP ViT-B/32 model...")
print("This will download ~350MB. Please ensure you have internet connection.")

try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    print("✅ Model downloaded successfully!")
    print(f"   Device: {device}")
    print(f"   Model cached at: ~/.cache/clip/ViT-B-32.pt")
    print("\nYou can now run the server offline!")
except Exception as e:
    print(f"❌ Error downloading model: {e}")
    print("\nTroubleshooting:")
    print("1. Check your internet connection")
    print("2. Try using a VPN if the download is blocked")
    print("3. Or manually download from: https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt")
    print("4. Save it to: ~/.cache/clip/ViT-B-32.pt") 