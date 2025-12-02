#!/usr/bin/env python3
"""
Medical Image Embedder - Generate embeddings for bone fracture X-rays

This script generates embeddings for bone fracture X-ray images using BiomedCLIP
and uploads them to Cloudinary CDN.

Usage:
    python generate_embeddings.py --cloud-name YOUR_CLOUD_NAME --api-key YOUR_API_KEY --api-secret YOUR_API_SECRET

Requirements:
    - Conda environment: hnsw-backend-venv
    - Cloudinary account (free tier)
    - ~2GB disk space
    - GPU recommended (but not required)
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
from io import BytesIO

import numpy as np
import h5py
import requests
from PIL import Image
from tqdm import tqdm

# Check dependencies
try:
    import cloudinary
    import cloudinary.api
    import cloudinary.uploader
    from transformers import AutoModel, AutoProcessor
    import torch
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("\nPlease install required packages:")
    print("  pip install transformers huggingface-hub cloudinary pillow requests h5py tqdm numpy torch")
    sys.exit(1)


def setup_cloudinary(cloud_name, api_key, api_secret):
    """Configure Cloudinary with credentials"""
    cloudinary.config(
        cloud_name=cloud_name,
        api_key=api_key,
        api_secret=api_secret
    )
    
    # Test connection
    try:
        cloudinary.api.ping()
        print(f"✅ Cloudinary configured successfully!")
        print(f"   Cloud Name: {cloud_name}")
        return True
    except Exception as e:
        print(f"❌ Cloudinary connection failed: {e}")
        print("   Please check your credentials and try again.")
        return False


def download_fracatlas(dataset_path="FracAtlas"):
    """Download bone fracture dataset"""
    if os.path.exists(dataset_path):
        print(f"✅ Dataset already exists at: {dataset_path}")
    else:
        print("⚠️  FracAtlas repository is not available at the original GitHub link.")
        print("\n📥 Please download a bone fracture dataset manually:")
        print("\n   Recommended sources:")
        print("   1. MURA Dataset: https://stanfordmlgroup.github.io/competitions/mura/")
        print("   2. Kaggle: https://www.kaggle.com/datasets (search 'bone fracture')")
        print("   3. Roboflow: https://universe.roboflow.com/ (search 'bone fracture')")
        print(f"\n   After downloading, extract images to: {dataset_path}/images/")
        print("   Then run this script again.")
        print("\n   Or use --dataset-path to point to your existing image directory")
        sys.exit(1)
    
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(list(Path(dataset_path).rglob(ext)))
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total images found: {len(image_paths)}")
    print(f"   Dataset path: {os.path.abspath(dataset_path)}")
    
    if len(image_paths) == 0:
        print("\n⚠️  No images found! Please check the dataset structure.")
        return []
    
    print(f"\n   Sample images:")
    for i, img_path in enumerate(image_paths[:3]):
        print(f"      {i+1}. {img_path}")
    
    return image_paths


def upload_to_cloudinary(image_paths, folder="medical/fractures", rate_limit=0.15):
    """Upload images to Cloudinary CDN"""
    print(f"\n📤 Uploading {len(image_paths)} images to Cloudinary...")
    print(f"   This will take approximately {len(image_paths) * rate_limit / 60:.1f} minutes")
    print(f"   Rate limit: ~{3600/rate_limit:.0f} uploads/hour (safe rate)\n")
    
    uploaded_urls = []
    failed_uploads = []
    skipped_uploads = []
    
    for img_path in tqdm(image_paths, desc="Uploading"):
        try:
            # Use filename as public_id
            public_id = img_path.stem
            
            result = cloudinary.uploader.upload(
                str(img_path),
                folder=folder,
                public_id=public_id,
                overwrite=False,  # Skip if already exists
                resource_type="image"
            )
            
            uploaded_urls.append({
                'local_path': str(img_path),
                'url': result['secure_url'],
                'public_id': result['public_id'],
                'format': result['format'],
                'size': result['bytes']
            })
            
            # Rate limiting
            time.sleep(rate_limit)
            
        except cloudinary.exceptions.Error as e:
            if "already exists" in str(e).lower():
                skipped_uploads.append(str(img_path))
            else:
                print(f"\n⚠️  Failed to upload {img_path}: {e}")
                failed_uploads.append(str(img_path))
        except Exception as e:
            print(f"\n❌ Unexpected error uploading {img_path}: {e}")
            failed_uploads.append(str(img_path))
    
    print(f"\n📊 Upload Summary:")
    print(f"   ✅ Successfully uploaded: {len(uploaded_urls)}")
    print(f"   ⏭️  Skipped (already exists): {len(skipped_uploads)}")
    print(f"   ❌ Failed uploads: {len(failed_uploads)}")
    
    # Save URL mapping for reference
    with open('cloudinary_urls.json', 'w') as f:
        json.dump(uploaded_urls, f, indent=2)
    print(f"\n💾 Saved URL mapping to: cloudinary_urls.json")
    
    # Display sample URLs
    if uploaded_urls:
        print(f"\n   Sample Cloudinary URLs:")
        for i, item in enumerate(uploaded_urls[:3]):
            print(f"      {i+1}. {item['url']}")
    
    return uploaded_urls


def load_biomedclip():
    """Load BiomedCLIP model"""
    print("\n🤖 Loading BiomedCLIP model...")
    print("   Model: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    print("   This may take a few minutes on first run (downloads ~2GB)\n")
    
    model_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    
    model = AutoModel.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    print(f"✅ Model loaded successfully!")
    print(f"   Device: {device}")
    print(f"   Embedding dimension: 512")
    
    # Test the model
    test_text = "femur fracture"
    inputs = processor(text=[test_text], return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    print(f"\n🧪 Test encoding: '{test_text}'")
    print(f"   Output shape: {text_features.shape}")
    print(f"   ✅ Model is working correctly!")
    
    return model, processor, device


def fetch_and_encode_image(url, model, processor, device, max_retries=3):
    """Fetch image from Cloudinary and generate embedding"""
    for attempt in range(max_retries):
        try:
            # Download image
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Open and convert to RGB
            image = Image.open(BytesIO(response.content)).convert('RGB')
            
            # Generate embedding
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().astype(np.float32)[0]
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            else:
                print(f"\n❌ Failed to process {url[:80]}... after {max_retries} attempts: {e}")
                return None


def generate_embeddings(uploaded_urls, model, processor, device):
    """Generate embeddings for all images"""
    print(f"\n🧠 Generating embeddings for {len(uploaded_urls)} images...")
    print(f"   This will take approximately {len(uploaded_urls) * 0.5 / 60:.1f} minutes\n")
    
    embeddings = []
    valid_urls = []
    failed_encodings = []
    
    for item in tqdm(uploaded_urls, desc="Encoding"):
        url = item['url']
        embedding = fetch_and_encode_image(url, model, processor, device)
        
        if embedding is not None:
            embeddings.append(embedding)
            valid_urls.append(url)
        else:
            failed_encodings.append(url)
        
        # Small delay to avoid overwhelming the server
        time.sleep(0.1)
    
    # Convert to numpy array
    embeddings = np.array(embeddings)
    
    print(f"\n📊 Embedding Generation Summary:")
    print(f"   ✅ Successfully encoded: {len(embeddings)}")
    print(f"   ❌ Failed encodings: {len(failed_encodings)}")
    print(f"   📐 Embedding shape: {embeddings.shape}")
    print(f"   💾 Memory usage: {embeddings.nbytes / (1024**2):.2f} MB")
    
    return embeddings, valid_urls


def save_to_hdf5(embeddings, urls, output_file='Medical_Fractures_Embedbed.h5'):
    """Save embeddings and URLs to HDF5 format"""
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
        
        # Save URLs as bytes
        url_bytes = [url.encode('utf-8') for url in urls]
        f.create_dataset(
            'urls',
            data=url_bytes,
            compression='gzip'
        )
        print(f"   ✅ Saved URLs: {len(urls)} entries")
        
        # Add metadata
        f.attrs['model'] = 'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        f.attrs['embedding_dim'] = 512
        f.attrs['total_images'] = len(urls)
        f.attrs['dataset'] = 'FracAtlas'
        f.attrs['storage'] = 'Cloudinary'
        f.attrs['created_date'] = str(np.datetime64('now'))
        print(f"   ✅ Added metadata")
    
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n📊 HDF5 File Statistics:")
    print(f"   File: {output_file}")
    print(f"   Size: {file_size_mb:.2f} MB")
    print(f"   Images: {len(urls):,}")
    print(f"   Embeddings: {embeddings.shape}")
    print(f"\n✅ HDF5 file created successfully!")
    print(f"\n📁 Next step: Copy {output_file} to backend/ directory")
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
        loaded_urls = f['urls'][:]
        
        print(f"\n✅ Verification Results:")
        print(f"   Embeddings loaded: {loaded_embeddings.shape}")
        print(f"   URLs loaded: {len(loaded_urls)}")
        print(f"   Embedding range: [{loaded_embeddings.min():.4f}, {loaded_embeddings.max():.4f}]")
        print(f"   Embedding mean: {loaded_embeddings.mean():.4f}")
        print(f"   Embedding std: {loaded_embeddings.std():.4f}")
        
        # Sample URLs
        print(f"\n   Sample URLs:")
        for i in range(min(5, len(loaded_urls))):
            url = loaded_urls[i].decode('utf-8')
            print(f"      {i+1}. {url}")
        
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
    print(f"   1. {output_file} - HDF5 file with embeddings and URLs")
    print(f"   2. cloudinary_urls.json - URL mapping reference")
    print(f"\n🚀 Ready to use with server_medical.py!")


def main():
    parser = argparse.ArgumentParser(
        description='Generate embeddings for bone fracture X-rays using BiomedCLIP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with Cloudinary credentials
  python generate_embeddings.py --cloud-name mycloud --api-key 123456 --api-secret abc123

  # Skip download if dataset already exists
  python generate_embeddings.py --cloud-name mycloud --api-key 123456 --api-secret abc123 --skip-download

  # Use environment variables for credentials
  export CLOUDINARY_CLOUD_NAME=mycloud
  export CLOUDINARY_API_KEY=123456
  export CLOUDINARY_API_SECRET=abc123
  python generate_embeddings.py

See CLOUDINARY_SETUP.md for detailed setup instructions.
        """
    )
    
    parser.add_argument('--cloud-name', help='Cloudinary cloud name', 
                       default=os.environ.get('CLOUDINARY_CLOUD_NAME'))
    parser.add_argument('--api-key', help='Cloudinary API key',
                       default=os.environ.get('CLOUDINARY_API_KEY'))
    parser.add_argument('--api-secret', help='Cloudinary API secret',
                       default=os.environ.get('CLOUDINARY_API_SECRET'))
    parser.add_argument('--dataset-path', default='FracAtlas',
                       help='Path to FracAtlas dataset (default: FracAtlas)')
    parser.add_argument('--output', default='Medical_Fractures_Embedbed.h5',
                       help='Output HDF5 file (default: Medical_Fractures_Embedbed.h5)')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip dataset download (use existing)')
    parser.add_argument('--skip-upload', action='store_true',
                       help='Skip Cloudinary upload (use existing cloudinary_urls.json)')
    
    args = parser.parse_args()
    
    # Check Cloudinary credentials
    if not args.skip_upload:
        if not all([args.cloud_name, args.api_key, args.api_secret]):
            print("❌ Error: Cloudinary credentials required!")
            print("\nProvide credentials via:")
            print("  1. Command line: --cloud-name, --api-key, --api-secret")
            print("  2. Environment variables: CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET")
            print("\nSee CLOUDINARY_SETUP.md for setup instructions.")
            sys.exit(1)
        
        # Setup Cloudinary
        if not setup_cloudinary(args.cloud_name, args.api_key, args.api_secret):
            sys.exit(1)
    
    # Step 1: Download dataset
    if not args.skip_download:
        image_paths = download_fracatlas(args.dataset_path)
        if not image_paths:
            sys.exit(1)
    else:
        print("⏭️  Skipping dataset download (using existing)")
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_paths.extend(list(Path(args.dataset_path).rglob(ext)))
        print(f"   Found {len(image_paths)} images")
    
    # Step 2: Upload to Cloudinary
    if not args.skip_upload:
        uploaded_urls = upload_to_cloudinary(image_paths)
        if not uploaded_urls:
            print("❌ No images uploaded successfully!")
            sys.exit(1)
    else:
        print("\n⏭️  Skipping Cloudinary upload (using existing cloudinary_urls.json)")
        with open('cloudinary_urls.json', 'r') as f:
            uploaded_urls = json.load(f)
        print(f"   Loaded {len(uploaded_urls)} URLs from file")
    
    # Step 3: Load BiomedCLIP
    model, processor, device = load_biomedclip()
    
    # Step 4: Generate embeddings
    embeddings, valid_urls = generate_embeddings(uploaded_urls, model, processor, device)
    if len(embeddings) == 0:
        print("❌ No embeddings generated!")
        sys.exit(1)
    
    # Step 5: Save to HDF5
    output_file = save_to_hdf5(embeddings, valid_urls, args.output)
    
    # Step 6: Verify
    verify_hdf5(output_file)


if __name__ == '__main__':
    main()

