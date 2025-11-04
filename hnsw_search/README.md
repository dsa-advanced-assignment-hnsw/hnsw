# HNSW Search: Efficient k-NN Search for Images and Papers

This repository contains a Jupyter Notebook (`hnsw.ipynb`) that demonstrates how to use the **HNSW (Hierarchical Navigable Small World)** algorithm via the `hnswlib` library to perform efficient **k-nearest neighbors (k-NN)** searches on large datasets of embeddings. The notebook supports both **image search** and **academic paper search** using pre-computed embeddings and indices.

## 📋 Table of Contents

- [Features](#features)
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Workflow](#workflow)
- [HNSW Algorithm](#hnsw-algorithm)
- [Configuration](#configuration)
- [Examples](#examples)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [File Structure](#file-structure)
- [Attribution and Licensing](#attribution-and-licensing)

## ✨ Features

- **Fast k-NN Search**: Performs efficient approximate nearest neighbor searches on millions of vectors
- **Dual Search Modes**: Supports both image search and paper search
- **Text-to-Embedding**: Converts text queries to embeddings using pre-trained models
- **Pre-built Indices**: Uses optimized HNSW indices for fast retrieval
- **Memory Efficient**: Loads indices on-demand and includes memory cleanup
- **Cosine Similarity**: Uses cosine similarity for semantic matching
- **Scalable**: Handles datasets with millions of embeddings efficiently

## 📖 Overview

The notebook demonstrates semantic search capabilities using:

1. **Image Search**: Find similar images based on text descriptions using CLIP embeddings
2. **Paper Search**: Find relevant academic papers based on text queries using Sentence Transformer embeddings

Both search modes leverage pre-computed embeddings stored in HDF5 format and optimized HNSW indices for fast retrieval.

## 📦 Prerequisites

- **Python**: 3.8 or higher
- **Jupyter Notebook** or **JupyterLab**
- **Internet Connection** (for downloading pre-computed embeddings and indices)
- **Sufficient Disk Space**: 
  - Image embeddings: ~100-500 MB (depends on dataset size)
  - Paper embeddings: ~500 MB - 2 GB (depends on dataset size)
  - HNSW indices: ~50-200 MB each
- **RAM**: Recommended 8GB+ (16GB+ for large datasets)
- **Git** (for cloning the EmbedX repository)

## 🚀 Installation

### Step 1: Navigate to the Directory

```bash
cd hnsw_search
```

### Step 2: Install Dependencies

The notebook will automatically install all required dependencies. However, you can also install them manually:

```bash
# Install core dependencies
pip install gdown hnswlib

# Clone and install EmbedX
git clone https://github.com/huynguyen6906/EmbedX.git
pip install -r EmbedX/requirements.txt
cd EmbedX && pip install . && cd ..
```

### Required Packages

- `hnswlib` - HNSW library for approximate nearest neighbor search
- `gdown` - Google Drive downloader
- `h5py` - HDF5 file handling for embeddings
- `embedx` - Custom embedding utilities (from EmbedX repository)
- `sentence-transformers` - For text embeddings (via EmbedX)
- `clip` - OpenAI CLIP for image-text embeddings (via EmbedX)
- `torch` - PyTorch for deep learning (via EmbedX)
- `numpy` - Numerical computing

## 📖 Usage

### Step-by-Step Execution

1. **Open the Notebook**
   ```bash
   jupyter notebook hnsw.ipynb
   ```

2. **Run Cells Sequentially**
   - **Cell 1**: Initialization and setup (installs dependencies, clones EmbedX)
   - **Cell 2**: Import required libraries
   - **Cell 3**: Download pre-computed embeddings and HNSW indices (if not cached)
   - **Cell 4**: Image search setup and execution
   - **Cell 5**: Paper search setup and execution

3. **Monitor Downloads**
   - First run will download embeddings and indices (may take time)
   - Files are cached in `.cache/` directory for subsequent runs

### Quick Start: Image Search

```python
# Example: Search for images related to "computer"
Text_queries = ["computer"]
queries = [embedx.image.embed_Text(Text_queries[0])]
indices, distances = image_search.knn_query(queries, k=10)

# Display results
for idx in indices[0]:
    print(image_urls[idx])
```

### Quick Start: Paper Search

```python
# Example: Search for papers related to "machine learning"
Text_queries = ["machine learning"]
queries = [embedx.text.embed_Text(Text_queries[0])]
indices, distances = paper_search.knn_query(queries, k=10)

# Display results
for idx in indices[0]:
    print(paper_urls[idx])
```

## 🔄 Workflow

### 1. Setup & Initialization
- Installs `hnswlib` and `gdown`
- Clones and installs EmbedX repository
- Sets up the working environment

### 2. Download Cache Files
Downloads pre-computed files if not already present:

**Image Search Files**:
- `Image_Embedded.h5`: Image embeddings (CLIP vectors)
- `hnsw_image_index.bin`: Pre-built HNSW index for images
- `hnsw_image_index.json`: Index configuration (dim, max_element, ef)

**Paper Search Files**:
- `Papers_Embedbed_0-1000000.h5`: Paper embeddings (Sentence Transformer vectors)
- `hnsw_paper_index.bin`: Pre-built HNSW index for papers
- `hnsw_paper_index.json`: Index configuration (dim, max_element, ef)

### 3. Image Search Pipeline

1. **Load Image Data**:
   - Reads image URLs and embeddings from HDF5 file
   - Embeddings are 512-dimensional CLIP vectors

2. **Load HNSW Index**:
   - Reads index configuration from JSON file
   - Initializes HNSW index with cosine similarity
   - Loads pre-built index from binary file

3. **Perform Search**:
   - Converts text query to embedding using `embedx.image.embed_Text()`
   - Performs k-NN search with specified `k` (default: 10)
   - Returns indices and distances for top-k results

4. **Display Results**:
   - Retrieves image URLs using result indices
   - Optionally displays distances for relevance scoring

### 4. Paper Search Pipeline

1. **Load Paper Data**:
   - Reads paper URLs and embeddings from HDF5 file
   - Embeddings are 1024-dimensional Sentence Transformer vectors

2. **Load HNSW Index**:
   - Reads index configuration from JSON file
   - Initializes HNSW index with cosine similarity
   - Loads pre-built index from binary file

3. **Perform Search**:
   - Converts text query to embedding using `embedx.text.embed_Text()`
   - Performs k-NN search with specified `k` (default: 10)
   - Returns indices and distances for top-k results

4. **Display Results**:
   - Retrieves arXiv paper URLs using result indices
   - Optionally displays distances for relevance scoring

### 5. Memory Cleanup
- Deletes loaded data and indices after use
- Frees up memory for subsequent operations

## 🔍 HNSW Algorithm

### What is HNSW?

**Hierarchical Navigable Small World (HNSW)** is an approximate nearest neighbor search algorithm that provides:

- **Fast Search**: O(log N) search complexity
- **High Accuracy**: 95%+ recall on nearest neighbor queries
- **Memory Efficient**: Compact index representation
- **Scalable**: Handles millions to billions of vectors

### Key Parameters

- **`dim`**: Embedding dimension (512 for images, 1024 for papers)
- **`max_element`**: Maximum number of elements in the index
- **`ef`** (ef_search): Size of the candidate list during search (higher = more accurate but slower)
- **`space`**: Distance metric (`'cosine'` for normalized embeddings)

### How It Works

1. **Index Building**: Constructs a multi-layer graph structure
2. **Query Processing**: Navigates the graph from top to bottom layers
3. **Result Retrieval**: Returns k nearest neighbors based on cosine similarity

## ⚙️ Configuration

### Index Configuration Files

The JSON configuration files contain:

```json
[
  <dimension>,      // Embedding dimension
  <max_elements>,   // Maximum number of vectors in index
  <ef_search>       // Search parameter (candidate list size)
]
```

### Adjusting Search Parameters

To modify search behavior, adjust the `ef` parameter:

```python
# More accurate but slower
image_search.set_ef(200)

# Faster but less accurate
image_search.set_ef(50)
```

### Changing k (Number of Results)

```python
# Get top 20 results instead of default 10
indices, distances = image_search.knn_query(queries, k=20)
```

## 💡 Examples

### Example 1: Multi-Query Image Search

```python
# Search for multiple concepts
Text_queries = ["computer", "chemical", "animal"]
queries = [embedx.image.embed_Text(query) for query in Text_queries]
indices, distances = image_search.knn_query(queries, k=10)

# Display results for each query
for i, query in enumerate(Text_queries):
    print(f"Query: {query}")
    for idx in indices[i]:
        print(f"  {image_urls[idx]}")
    print()
```

### Example 2: Paper Search with Distance Scores

```python
# Search with distance information
Text_queries = ["deep learning", "neural networks"]
queries = [embedx.text.embed_Text(query) for query in Text_queries]
indices, distances = paper_search.knn_query(queries, k=5)

# Display results with similarity scores
for i, query in enumerate(Text_queries):
    print(f"Query: {query}")
    for j, idx in enumerate(indices[i]):
        similarity = 1 - distances[i][j]  # Convert distance to similarity
        print(f"  [{similarity:.3f}] {paper_urls[idx]}")
    print()
```

### Example 3: Building Custom Index

If you need to build your own HNSW index:

```python
import hnswlib
import numpy as np

# Initialize index
dim = 512  # Embedding dimension
num_elements = 10000  # Number of vectors
index = hnswlib.Index(space='cosine', dim=dim)
index.init_index(max_elements=num_elements, ef_construction=200, M=16)

# Add vectors
index.add_items(embeddings, ids)

# Save index
index.save_index("my_index.bin")

# Load and use
index.load_index("my_index.bin", max_elements=num_elements)
index.set_ef(50)  # Set search parameter
```

## ⚡ Performance

### Search Speed

- **Small datasets (< 100K vectors)**: < 1ms per query
- **Medium datasets (100K - 1M vectors)**: 1-10ms per query
- **Large datasets (1M - 10M vectors)**: 10-100ms per query

### Memory Usage

- **Index Memory**: ~10-20 bytes per vector
- **Embeddings Memory**: ~2-4 KB per vector (depending on dimension)
- **Total**: ~2-4 MB per 1000 vectors

### Accuracy vs Speed Trade-off

| ef Value | Recall | Search Time |
|----------|--------|-------------|
| 50 | ~90% | Fast |
| 100 | ~95% | Medium |
| 200 | ~98% | Slower |
| 500 | ~99% | Slow |

## 🐛 Troubleshooting

### Issue: Download Fails or is Slow

**Solution**:
- Check internet connection
- Files are large (especially embeddings), allow sufficient time
- Files are cached after first download

### Issue: Index File Not Found

**Solution**:
- Ensure all cache files were downloaded successfully
- Check `.cache/` directory for presence of `.bin` and `.json` files
- Re-run the download cell if files are missing

### Issue: Memory Errors

**Solution**:
- Process queries in smaller batches
- Delete variables after use (notebook includes cleanup cells)
- Reduce `ef` parameter to use less memory
- Close other applications to free RAM

### Issue: Import Errors

**Solution**:
```bash
# Reinstall dependencies
pip install --upgrade hnswlib gdown h5py

# Reinstall EmbedX
cd EmbedX
pip install -r requirements.txt
pip install .
```

### Issue: Embedding Generation Fails

**Solution**:
- Ensure EmbedX is properly installed
- Check that CLIP/Sentence Transformer models can be loaded
- Verify internet connection for model downloads

### Issue: Search Results Seem Inaccurate

**Solution**:
- Increase `ef` parameter for better accuracy
- Verify embeddings are L2-normalized (required for cosine similarity)
- Check that query text is meaningful and clear

### Issue: Slow Search Performance

**Solution**:
- Reduce `ef` parameter for faster search
- Reduce `k` (number of results) if not needed
- Ensure you're using GPU-accelerated embedding generation (if available)
- Use SSD instead of HDD for faster file I/O

## 📁 File Structure

```
hnsw_search/
├── hnsw.ipynb                    # Main notebook
├── README.md                     # This file
├── .cache/                       # Cache directory (created automatically)
│   ├── Image_Embedded.h5         # Image embeddings
│   ├── hnsw_image_index.bin      # Pre-built image HNSW index
│   ├── hnsw_image_index.json     # Image index configuration
│   ├── Papers_Embedbed_0-1000000.h5  # Paper embeddings
│   ├── hnsw_paper_index.bin      # Pre-built paper HNSW index
│   └── hnsw_paper_index.json     # Paper index configuration
└── EmbedX/                       # Cloned repository (created automatically)
    └── ...
```

## 🔗 Related Resources

- [HNSWlib Documentation](https://github.com/nmslib/hnswlib)
- [HNSW Algorithm Paper](https://arxiv.org/abs/1603.09320)
- [EmbedX Repository](https://github.com/huynguyen6906/EmbedX)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [Sentence Transformers](https://www.sbert.net/)

## 📝 Notes

- **First Run**: Will take longer due to downloading embeddings and indices
- **Subsequent Runs**: Uses cached files for faster startup
- **Embedding Models**: Uses CLIP for images and Sentence Transformers for papers
- **Normalization**: All embeddings are L2-normalized for cosine similarity
- **Memory Management**: Notebook includes cleanup cells to free memory
- **Scalability**: HNSW can handle datasets with millions of vectors efficiently

## 🎯 Use Cases

This notebook is ideal for:

- **Semantic Image Search**: Find images based on text descriptions
- **Academic Paper Discovery**: Find relevant papers based on queries
- **Content Recommendation**: Recommend similar content to users
- **Duplicate Detection**: Find similar items in large datasets
- **Clustering**: Group similar items together
- **Anomaly Detection**: Find items that are dissimilar to others

## 🔬 Technical Details

### Embedding Models

**Image Embeddings**:
- Model: CLIP ViT-B/32
- Dimension: 512
- Normalization: L2-normalized

**Paper Embeddings**:
- Model: all-roberta-large-v1 (Sentence Transformer)
- Dimension: 1024
- Normalization: L2-normalized

### Distance Metric

Both indices use **cosine similarity** (equivalent to cosine distance on normalized vectors), which is ideal for semantic similarity search.

### Index Construction

Indices are pre-built and optimized for:
- Fast search performance
- High recall accuracy
- Memory efficiency
- Scalability to millions of vectors

---

## ⚖️ Attribution and Licensing

### Project Code
The code in this repository (specifically `hnsw.ipynb`) is typically licensed under a standard open-source license such as the **MIT License**.

### Third-Party Assets

This project relies on assets from other sources, which are subject to their original licenses:

- **HNSWlib**: Licensed under **Apache 2.0** ([nmslib/hnswlib](https://github.com/nmslib/hnswlib))

- **EmbedX**: The EmbedX repository and its dependencies are subject to their respective licenses as specified in their repository.

- **Pre-computed Embeddings**: The embeddings and indices are generated from:
  - **Open Images V7**: Images licensed under **CC BY 4.0**
  - **arXiv Papers**: Metadata in public domain (CC0)

- **Models**: 
  - **CLIP**: Licensed under **MIT License** (OpenAI)
  - **Sentence Transformers**: Licensed under **MIT** or **Apache 2.0**

---

**Last Updated**: 2025
