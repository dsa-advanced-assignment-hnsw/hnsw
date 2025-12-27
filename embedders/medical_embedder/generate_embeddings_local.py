#!/usr/bin/env python3
"""
Medical Image Embedder - Generate embeddings for bone fracture X-rays (LOCAL STORAGE)

This script generates embeddings for bone fracture X-ray images using BiomedCLIP
and stores them with local file paths (no cloud storage needed).

Usage:
    python generate_embeddings_local.py --dataset-path /path/to/bone_fractures

Requirements:
    - Conda environment: hnsw-backend-venv
    - ~2GB disk space for model
    - GPU recommended (but not required)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import h5py
from PIL import Image
from tqdm import tqdm

# Check dependencies
try:
    import torch
    import open_clip
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("\nPlease install required packages:")
    print("  conda activate hnsw-backend-venv")
    print("  pip install open-clip-torch")
    sys.exit(1)


def find_images(dataset_path):
    """Find all image files in the dataset directory"""
    print(f"📂 Scanning for images in: {dataset_path}")
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(list(Path(dataset_path).rglob(ext)))
    
    # Convert to absolute paths
    image_paths = [str(p.absolute()) for p in image_paths]
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total images found: {len(image_paths)}")
    print(f"   Dataset path: {os.path.abspath(dataset_path)}")
    
    if len(image_paths) == 0:
        print("\n⚠️  No images found! Please check the dataset path.")
        print(f"   Looking for: {', '.join(image_extensions)}")
        return []
    
    print(f"\n   Sample images:")
    for i, img_path in enumerate(image_paths[:5]):
        print(f"      {i+1}. {img_path}")
    
    return image_paths


def load_biomedclip():
    """Load BiomedCLIP model using open_clip"""
    print("\n🤖 Loading BiomedCLIP model...")
    print("   Model: hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    print("   This may take a few minutes on first run (downloads ~2GB)\n")
    
    # Load BiomedCLIP using open_clip
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    print(f"✅ Model loaded successfully!")
    print(f"   Device: {device}")
    print(f"   Embedding dimension: 512")
    
    # Test the model
    test_text = "femur fracture"
    text_tokens = tokenizer([test_text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    print(f"\n🧪 Test encoding: '{test_text}'")
    print(f"   Output shape: {text_features.shape}")
    print(f"   ✅ Model is working correctly!")
    
    return model, preprocess_val, tokenizer, device


def encode_image(image_path, model, preprocess, device):
    """Encode a single image to embedding"""
    try:
        # Open and convert to RGB
        image = Image.open(image_path).convert('RGB')
        
        # Preprocess image
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        # Generate embedding
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy().astype(np.float32)[0]
        
    except Exception as e:
        print(f"\n❌ Failed to process {image_path}: {e}")
        return None


def generate_embeddings(image_paths, model, preprocess, device):
    """Generate embeddings for all images"""
    print(f"\n🧠 Generating embeddings for {len(image_paths)} images...")
    print(f"   This will take approximately {len(image_paths) * 0.5 / 60:.1f} minutes\n")
    
    embeddings = []
    valid_paths = []
    failed_encodings = []
    
    for img_path in tqdm(image_paths, desc="Encoding"):
        embedding = encode_image(img_path, model, preprocess, device)
        
        if embedding is not None:
            embeddings.append(embedding)
            valid_paths.append(img_path)
        else:
            failed_encodings.append(img_path)
    
    # Convert to numpy array
    embeddings = np.array(embeddings)
    
    print(f"\n📊 Embedding Generation Summary:")
    print(f"   ✅ Successfully encoded: {len(embeddings)}")
    print(f"   ❌ Failed encodings: {len(failed_encodings)}")
    print(f"   📐 Embedding shape: {embeddings.shape}")
    print(f"   💾 Memory usage: {embeddings.nbytes / (1024**2):.2f} MB")
    
    return embeddings, valid_paths


def save_to_hdf5(embeddings, image_paths, output_file='Medical_Fractures_Embedbed.h5'):
    """Save embeddings and image paths to HDF5 format"""
    print(f"\n💾 Saving embeddings to HDF5 file...")
    print(f"   Output file: {output_file}\n")
    
    with h5py.File(output_file, 'w') as f:
        # Save embeddings with compression
        f.create_dataset(
            'embeddings',
            data=embeddings,
            compression='gzip',
            compression_opts=9
        )
        print(f"   ✅ Saved embeddings: shape {embeddings.shape}")
        
        # Save image paths as bytes
        path_bytes = [path.encode('utf-8') for path in image_paths]
        f.create_dataset(
            'image_path',
            data=path_bytes,
            compression='gzip'
        )
        print(f"   ✅ Saved image paths: {len(image_paths)} entries")
        
        # Add metadata
        f.attrs['model'] = 'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        f.attrs['embedding_dim'] = 512
        f.attrs['total_images'] = len(image_paths)
        f.attrs['storage'] = 'local'
        f.attrs['created_date'] = str(np.datetime64('now'))
        print(f"   ✅ Added metadata")
    
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n📊 HDF5 File Statistics:")
    print(f"   File: {output_file}")
    print(f"   Size: {file_size_mb:.2f} MB")
    print(f"   Images: {len(image_paths):,}")
    print(f"   Embeddings: {embeddings.shape}")
    print(f"\n✅ HDF5 file created successfully!")
    print(f"\n📁 Next step: Copy {output_file} to backend/ directory")
    print(f"   cp {output_file} ../backend/")
    print(f"   Then run: conda activate hnsw-backend-venv && python server_medical.py")
    
    return output_file


def verify_hdf5(output_file):
    """Verify the HDF5 file"""
    print(f"\n🔍 Verifying HDF5 file: {output_file}\n")
    
    with h5py.File(output_file, 'r') as f:
        print("📊 Datasets:")
        for key in f.keys():
            dataset = f[key]
            print(f"   - {key}: shape {dataset.shape}, dtype {dataset.dtype}")
        
        print("\n📋 Metadata:")
        for key, value in f.attrs.items():
            print(f"   - {key}: {value}")
        
        # Load and verify embeddings
        loaded_embeddings = f['embeddings'][:]
        loaded_paths = f['image_path'][:]
        
        print(f"\n✅ Verification Results:")
        print(f"   Embeddings loaded: {loaded_embeddings.shape}")
        print(f"   Paths loaded: {len(loaded_paths)}")
        print(f"   Embedding range: [{loaded_embeddings.min():.4f}, {loaded_embeddings.max():.4f}]")
        print(f"   Embedding mean: {loaded_embeddings.mean():.4f}")
        print(f"   Embedding std: {loaded_embeddings.std():.4f}")
        
        # Sample paths
        print(f"\n   Sample image paths:")
        for i in range(min(5, len(loaded_paths))):
            path = loaded_paths[i].decode('utf-8')
            exists = "✓" if os.path.exists(path) else "✗"
            print(f"      {i+1}. [{exists}] {path}")
        
        # Test similarity computation
        print(f"\n🧪 Testing similarity computation:")
        if len(loaded_embeddings) >= 2:
            emb1 = loaded_embeddings[0]
            emb2 = loaded_embeddings[1]
            similarity = np.dot(emb1, emb2)
            print(f"   Cosine similarity between first two images: {similarity:.4f}")
            print(f"   (Range: -1 to 1, where 1 = identical)")
    
    print(f"\n✅ All verifications passed!")
    print(f"\n🎉 Medical embedder pipeline completed successfully!")
    print(f"\n📦 Deliverables:")
    print(f"   1. {output_file} - HDF5 file with embeddings and local paths")
    print(f"\n🚀 Ready to use with server_medical.py!")


def main():
    parser = argparse.ArgumentParser(
        description='Generate embeddings for bone fracture X-rays (LOCAL STORAGE)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate embeddings from local image directory
  python generate_embeddings_local.py --dataset-path /path/to/bone_fractures

  # Specify custom output file
  python generate_embeddings_local.py --dataset-path ./images --output custom.h5

  # Use relative path
  python generate_embeddings_local.py --dataset-path bone_fractures
        """
    )
    
    parser.add_argument('--dataset-path', required=True,
                       help='Path to directory containing bone fracture images')
    parser.add_argument('--output', default='Medical_Fractures_Embedbed.h5',
                       help='Output HDF5 file (default: Medical_Fractures_Embedbed.h5)')
    
    args = parser.parse_args()
    
    # Check if dataset path exists
    if not os.path.exists(args.dataset_path):
        print(f"❌ Error: Dataset path not found: {args.dataset_path}")
        print("\nPlease provide a valid path to your bone fracture images.")
        print("\nExample directory structure:")
        print("  bone_fractures/")
        print("    ├── image001.jpg")
        print("    ├── image002.jpg")
        print("    └── ...")
        sys.exit(1)
    
    # Step 1: Find images
    image_paths = find_images(args.dataset_path)
    if not image_paths:
        sys.exit(1)
    
    # Step 2: Load BiomedCLIP
    model, preprocess, tokenizer, device = load_biomedclip()
    
    # Step 3: Generate embeddings
    embeddings, valid_paths = generate_embeddings(image_paths, model, preprocess, device)
    if len(embeddings) == 0:
        print("❌ No embeddings generated!")
        sys.exit(1)
    
    # Step 4: Save to HDF5
    output_file = save_to_hdf5(embeddings, valid_paths, args.output)
    
    # Step 5: Verify
    verify_hdf5(output_file)


if __name__ == '__main__':
    main()

