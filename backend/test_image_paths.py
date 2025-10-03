#!/usr/bin/env python3
"""
Test script to verify image path resolution
"""
import os
import h5py

print("🔍 Testing Image Path Configuration\n")

# Get base directory (where this script is)
base_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Backend directory: {base_dir}")

# Check if images folder exists
images_dir = os.path.join(base_dir, "images")
if os.path.exists(images_dir):
    print(f"✅ Images directory exists: {images_dir}")
    
    # Count files in images folder
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'))]
    print(f"📁 Image files in directory: {len(image_files)}")
    if image_files:
        print(f"   Sample files: {image_files[:3]}")
else:
    print(f"⚠️  Images directory not found: {images_dir}")
    print(f"   Please create it with: mkdir -p {images_dir}")

# Check HDF5 file
h5_file = os.path.join(base_dir, "images_embeds.h5")
if os.path.exists(h5_file):
    print(f"\n✅ HDF5 file exists: {h5_file}")
    
    with h5py.File(h5_file, "r") as f:
        paths = f["image_path"][:]
        print(f"📊 Total images in index: {len(paths)}")
        
        # Show sample paths
        print(f"\n📝 Sample paths from HDF5:")
        for i, path in enumerate(paths[:3]):
            path_str = path.decode() if isinstance(path, bytes) else path
            print(f"   {i+1}. {path_str}")
            
            # Check if this would resolve correctly
            if path_str.startswith('./'):
                clean_path = path_str[2:]
            else:
                clean_path = path_str
            
            full_path = os.path.join(base_dir, clean_path)
            exists = "✅" if os.path.exists(full_path) else "❌"
            print(f"      → {exists} {full_path}")
        
        print(f"\n📋 Summary:")
        print(f"   - Paths in HDF5: {len(paths)}")
        print(f"   - Images in folder: {len(image_files) if os.path.exists(images_dir) else 0}")
        
        if os.path.exists(images_dir) and len(image_files) > 0:
            print(f"\n✅ Setup is ready! Images will display once you add all files.")
        else:
            print(f"\n⚠️  Action needed:")
            print(f"   1. Make sure images directory exists: {images_dir}")
            print(f"   2. Add {len(paths)} image files matching the paths in HDF5")
            print(f"   3. Example: Place 34894.jpg, 59708.jpg, etc. in {images_dir}/")
else:
    print(f"❌ HDF5 file not found: {h5_file}")

print("\n" + "="*60) 