# IMAGE EMBEDDER: CLIP Image Embedding Generator

This repository contains a Jupyter Notebook (`image_embedder.ipynb`) that implements a complete, fault-tolerant pipeline for generating image embeddings using OpenAI's CLIP model. The pipeline downloads images from the **Open Images V7** dataset, processes them using **CLIP (ViT-B/32)**, and stores the resulting URL-vector pairs in HDF5 files for subsequent use in semantic image search or recommendation systems.

## 📋 Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Workflow](#workflow)
- [Output Format](#output-format)
- [Model Information](#model-information)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Attribution and Licensing](#attribution-and-licensing)

## ✨ Features

- **Automated Image Download**: Downloads images from Open Images V7 dataset using parallel processing
- **CLIP-based Embeddings**: Uses OpenAI's CLIP (ViT-B/32) model for state-of-the-art visual embeddings
- **Chunked Processing**: Processes large datasets in manageable chunks to optimize memory usage
- **GPU Acceleration**: Leverages GPU for fast batch encoding when available
- **Parallel Preprocessing**: Uses multi-threading for efficient image loading and preprocessing
- **L2 Normalization**: Automatically normalizes embeddings for efficient cosine similarity calculations
- **Fault Tolerant**: Processes data in chunks, allowing for recovery from interruptions
- **Efficient Storage**: Saves results in HDF5 format for fast I/O and compression
- **Automatic Cleanup**: Removes temporary images and chunk files after processing

## 📦 Prerequisites

- **Python**: 3.8 or higher
- **Jupyter Notebook** or **JupyterLab**
- **CUDA-capable GPU** (recommended for faster processing, but CPU is supported)
- **Internet Connection** (for downloading images and models)
- **Sufficient Disk Space**: 
  - Metadata files: ~100 MB
  - Images: Varies by chunk size (typically 10-50 GB for full dataset)
  - Embeddings: ~4-8 MB per 1000 images (512-dimensional vectors)
- **RAM**: Recommended 16GB+ (32GB+ for processing large batches)
- **wget**: Required for downloading the downloader script

## 🚀 Installation

### Step 1: Navigate to the Directory

```bash
cd image_embedder
```

### Step 2: Install Dependencies

The notebook will automatically install all required dependencies. However, you can also install them manually:

```bash
# Install all required packages including CLIP from OpenAI's GitHub
pip install requests boto3 h5py gdown torch tqdm typing numpy Pillow ipywidgets git+https://github.com/openai/CLIP.git
```

### Required Packages

- `torch` - PyTorch for deep learning
- `clip` - OpenAI CLIP model (installed from GitHub)
- `numpy` - Numerical computing
- `h5py` - HDF5 file handling
- `Pillow` - Image processing
- `tqdm` - Progress bars
- `gdown` - Google Drive downloader
- `requests` - HTTP library
- `boto3` - AWS SDK (used by downloader script)
- `ipywidgets` - Interactive widgets for Jupyter

## ⚙️ Configuration

Before running the notebook, configure the processing parameters in **Cell 16**:

```python
START = 0           # Starting image index (0-based)
END = n_images      # Ending image index (use n_images for all images)
CHUNK = 10000       # Number of images to process per chunk
GPU_BATCH_SIZE = 512  # Batch size for GPU encoding
NUM_WORKERS = 50    # Number of threads for parallel preprocessing
```

### Configuration Parameters

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `START` | First image index to process (0-based) | `0` |
| `END` | Last image index to process (exclusive) | `n_images` (all images) |
| `CHUNK` | Number of images processed per chunk | `10000` (adjust based on RAM) |
| `GPU_BATCH_SIZE` | Batch size for GPU encoding | `512` (reduce if OOM errors) |
| `NUM_WORKERS` | Thread pool size for image preprocessing | `50` (adjust based on CPU cores) |

### Recommended Settings by Dataset Size

**Small Dataset (< 50K images)**:
```python
CHUNK = 5000
GPU_BATCH_SIZE = 256
NUM_WORKERS = 25
```

**Medium Dataset (50K - 500K images)**:
```python
CHUNK = 10000
GPU_BATCH_SIZE = 512
NUM_WORKERS = 50
```

**Large Dataset (> 500K images)**:
```python
CHUNK = 20000
GPU_BATCH_SIZE = 512
NUM_WORKERS = 50
```

### GPU Memory Considerations

- **8GB GPU**: Use `GPU_BATCH_SIZE = 256`
- **16GB GPU**: Use `GPU_BATCH_SIZE = 512`
- **24GB+ GPU**: Use `GPU_BATCH_SIZE = 1024` or higher

**Note**: Adjust `CHUNK` based on available RAM. Larger chunks process faster but require more memory.

## 📖 Usage

### Step-by-Step Execution

1. **Open the Notebook**
   ```bash
   jupyter notebook image_embedder.ipynb
   ```

2. **Run Cells Sequentially**
   - **Cell 1**: Initialization and setup (installs dependencies including CLIP)
   - **Cell 2**: Import required libraries
   - **Cell 3**: Download `downloader.py` utility script
   - **Cell 4**: Download metadata files (image_ids.json, image_urls.json)
   - **Cell 5**: Load metadata and CLIP model
   - **Cell 6**: Define helper functions (`read_json`, `load_and_preprocess_image`, `merge_HDF5_files`)
   - **Cell 7**: Configure processing parameters
   - **Cell 8**: Main processing loop (downloads, preprocesses, encodes images in chunks)
   - **Cell 9**: Merge chunk files and cleanup

3. **Monitor Progress**
   - Progress bars show embedding progress for each chunk
   - Download progress is shown by the downloader script
   - Merge progress is displayed when combining files

### Example: Processing First 20,000 Images

```python
START = 0
END = 20000
CHUNK = 10000
GPU_BATCH_SIZE = 512
NUM_WORKERS = 50
```

This will:
- Process 10,000 images per chunk (2 chunks total)
- Create temporary files: `OUTPUT/Images_Embedded_0.h5` and `OUTPUT/Images_Embedded_1.h5`
- Merge into final file: `OUTPUT/Images_Embedded.h5`
- Delete temporary chunk files and downloaded images

## 🔄 Workflow

### 1. Setup & Initialization
- Installs all required Python packages
- Downloads the `downloader.py` script from Open Images toolkit
- Sets up the working environment

### 2. Data Preparation
- Creates `.cache` and `RAW_DATASET` directories
- Downloads `downloader.py` utility script
- Downloads metadata files:
  - `image_ids.json`: List of image IDs
  - `image_urls.json`: List of corresponding image URLs
- Both files are hosted on Google Drive

### 3. Model Loading
- Loads OpenAI CLIP (ViT-B/32) model
- Model is cached locally after first download
- Automatically uses GPU if available, otherwise falls back to CPU
- Embedding dimension: 512

### 4. Main Processing Loop (Chunking)
For each chunk:

1. **Image Download**: 
   - Creates a list file with image IDs for the current chunk
   - Uses `downloader.py` with 100 parallel processes to download images
   - Images are saved to `.cache/Images/` directory

2. **Image Preprocessing (CPU - Multi-threaded)**:
   - Uses `ThreadPoolExecutor` with `NUM_WORKERS` threads
   - Loads images from disk using PIL
   - Applies CLIP preprocessing pipeline (resize, normalize, etc.)
   - Returns preprocessed tensors on CPU

3. **Embedding Generation (GPU - Batch Processing)**:
   - Gathers preprocessed tensors into batches of size `GPU_BATCH_SIZE`
   - Moves batches to GPU for encoding
   - Uses `model.encode_image()` to generate embeddings
   - Applies L2 normalization for cosine similarity optimization

4. **Saving Chunk Results**:
   - Writes URLs and embeddings to temporary HDF5 chunk file
   - Format: `OUTPUT/Images_Embedded_{chunk_index}.h5`

5. **Cleanup**:
   - Deletes downloaded images from `.cache/Images/` to free disk space
   - Frees GPU memory before next chunk

### 5. Final Aggregation
- Merges all chunk files using `merge_HDF5_files` function
- Creates unified output: `OUTPUT/Images_Embedded.h5`
- Deletes temporary chunk files to free disk space

## 📊 Output Format

The final HDF5 file contains two datasets:

### Dataset Structure

```python
import h5py

with h5py.File('OUTPUT/Images_Embedded.h5', 'r') as f:
    urls = f['urls'][:]          # Array of image URLs (bytes)
    embeddings = f['embeddings'][:] # Array of normalized embeddings (float32)
    
    print(f"Number of images: {len(urls)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"URL dtype: {urls.dtype}")
    print(f"Embedding dtype: {embeddings.dtype}")
```

### Output Details

| Dataset | Type | Shape | Description |
|---------|------|-------|-------------|
| `urls` | `bytes` | `(N,)` | Image URLs from Open Images V7 |
| `embeddings` | `float32` | `(N, 512)` | L2-normalized CLIP embeddings |

### Example: Loading and Using Output

```python
import h5py
import numpy as np

# Load the embeddings
with h5py.File('OUTPUT/Images_Embedded.h5', 'r') as f:
    urls = [url.decode('utf-8') for url in f['urls'][:]]
    embeddings = f['embeddings'][:]

# Example: Find most similar images
query_embedding = embeddings[0]  # Use first image as query
similarities = np.dot(embeddings, query_embedding)  # Cosine similarity (L2-normalized)
top_indices = np.argsort(similarities)[::-1][:10]  # Top 10 most similar

for idx in top_indices:
    print(f"Similarity: {similarities[idx]:.4f}, URL: {urls[idx]}")
```

## 🤖 Model Information

### Model: CLIP ViT-B/32

- **Architecture**: Vision Transformer (ViT) with B/32 configuration
- **Embedding Dimension**: 512
- **Input Size**: 224x224 pixels (automatically resized by preprocessor)
- **Performance**: State-of-the-art on visual similarity tasks
- **License**: MIT License (OpenAI)
- **Source**: [OpenAI CLIP](https://github.com/openai/CLIP)

### Model Characteristics

- Pre-trained on 400 million image-text pairs
- Optimized for visual understanding and semantic similarity
- Produces dense vector embeddings suitable for:
  - Image similarity search
  - Visual recommendation systems
  - Image clustering and classification
  - Cross-modal retrieval (image-text matching)

### CLIP Preprocessing Pipeline

The model automatically applies:
1. Image resizing to 224x224
2. Center cropping
3. Normalization using ImageNet statistics
4. Tensor conversion for PyTorch

## ⚡ Performance Optimization

### GPU vs CPU

- **GPU**: Significantly faster (10-50x speedup depending on GPU)
- **CPU**: Supported but much slower, recommended only for small datasets

### Batch Size Tuning

- Larger batches = faster processing but more GPU memory
- Start with `GPU_BATCH_SIZE = 512` and adjust based on GPU memory
- If you encounter OOM errors, reduce batch size

### Thread Workers

- More workers = faster image loading/preprocessing
- Recommended: `NUM_WORKERS = 50` for modern CPUs
- Adjust based on CPU cores (typically 2x number of cores)

### Chunk Size

- Larger chunks = less overhead but more memory usage
- Balance between RAM availability and processing efficiency
- Recommended: `CHUNK = 10000` for most systems

## 🐛 Troubleshooting

### Issue: Out of Memory (OOM) Errors

**Solution**: 
- Reduce `GPU_BATCH_SIZE` (e.g., from 512 to 256)
- Reduce `CHUNK` size (e.g., from 10000 to 5000)
- Close other GPU applications
- Process fewer images at once

### Issue: Download Fails or is Slow

**Solution**:
- Check internet connection
- Open Images dataset images are large, allow sufficient time
- Downloader uses 100 parallel processes by default
- Some images may fail to download (automatically skipped)

### Issue: CLIP Installation Fails

**Solution**:
```bash
# Install CLIP manually
pip install git+https://github.com/openai/CLIP.git

# Or install specific version
pip install git+https://github.com/openai/CLIP.git@dcba3cb
```

### Issue: GPU Not Detected

**Solution**:
- Verify CUDA installation: `nvidia-smi`
- Check PyTorch CUDA support: `torch.cuda.is_available()`
- Reinstall PyTorch with CUDA support if needed
- Notebook will automatically fall back to CPU

### Issue: Image Download Script Fails

**Solution**:
- Ensure `wget` is installed: `sudo apt install wget` (Linux)
- Check if downloader.py was downloaded correctly
- Verify internet connection
- Some images may be unavailable (automatically skipped)

### Issue: HDF5 File Corruption

**Solution**:
- Re-run the notebook from the failed chunk
- Delete corrupted chunk files manually
- Adjust `START` parameter to skip processed chunks
- Check disk space availability

### Issue: Slow Processing

**Solution**:
- Use GPU instead of CPU (10-50x speedup)
- Increase `GPU_BATCH_SIZE` if GPU memory allows
- Increase `NUM_WORKERS` for faster preprocessing
- Use SSD instead of HDD for faster I/O

### Issue: Too Many Open Files Error

**Solution**:
- Reduce `NUM_WORKERS` (e.g., from 50 to 25)
- Increase system file descriptor limit:
  ```bash
  ulimit -n 4096  # Linux/Mac
  ```

### Issue: Image Processing Errors

**Solution**:
- Corrupted or invalid images are automatically skipped
- Errors are logged but don't stop processing
- Check console output for specific error messages

## 📁 File Structure

```
image_embedder/
├── image_embedder.ipynb    # Main notebook
├── README.md                # This file
├── .cache/                  # Cache directory (created automatically)
│   ├── downloader.py        # Open Images downloader script
│   ├── list_images.txt      # Temporary image list (created during processing)
│   └── Images/              # Downloaded images (temporary, deleted after processing)
├── RAW_DATASET/             # Metadata directory (created automatically)
│   ├── image_ids.json       # Image IDs from Open Images V7
│   └── image_urls.json      # Image URLs from Open Images V7
└── OUTPUT/                  # Output directory (created automatically)
    ├── Images_Embedded_0.h5 # Temporary chunk files (deleted after merge)
    ├── Images_Embedded_1.h5
    └── Images_Embedded.h5   # Final merged output file
```

## ⚖️ Attribution and Licensing

### Project Code
The code in this repository (specifically `image_embedder.ipynb` and the `merge_HDF5_files` function) is typically licensed under a standard open-source license such as the **MIT License**.

### Third-Party Assets

This project relies on assets from other sources, which are subject to their original licenses:

- **Open Images V7**: The image metadata (IDs and URLs) and the `downloader.py` script are components of the Open Images V7 dataset. This data is licensed under the **CC BY 4.0 license**.

- **OpenAI CLIP**: The CLIP model and its pre-trained weights are (c) OpenAI and licensed under the **MIT License**.

## 📝 Notes

- The first run will take longer due to model download and image downloading
- Images are downloaded temporarily and deleted after processing each chunk
- Processing time depends heavily on hardware (GPU highly recommended)
- Embeddings are deterministic (same image = same embedding)
- Chunk processing allows resuming from interruptions
- Some images may fail to download (automatically skipped)
- The downloader script uses boto3 for AWS S3 access (images hosted on AWS)

## 🔗 Related Resources

- [OpenAI CLIP Repository](https://github.com/openai/CLIP)
- [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html)
- [Open Images Downloader](https://github.com/openimages/dataset)
- [HDF5 Documentation](https://www.h5py.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 🎯 Use Cases

This pipeline is ideal for:

- **Image Search Engines**: Build semantic image search systems
- **Visual Recommendation**: Recommend similar images to users
- **Image Clustering**: Group similar images together
- **Content-Based Retrieval**: Find images based on visual similarity
- **ML Model Training**: Generate training data for downstream tasks
- **Dataset Preparation**: Prepare embeddings for machine learning pipelines

---

**Last Updated**: 2025
