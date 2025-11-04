# PAPER EMBEDDER: arXiv Paper Embedding Generator

This repository contains a Jupyter Notebook (`paper_embedder.ipynb`) that automates the process of downloading the full **arXiv metadata dataset**, generating dense vector embeddings for paper abstracts using a **high-performance Sentence Transformer** model, and saving the results into a single, comprehensive HDF5 file for subsequent use in semantic search or recommendation systems.

## 📋 Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Workflow](#workflow)
- [Output Format](#output-format)
- [Model Information](#model-information)
- [Troubleshooting](#troubleshooting)
- [Attribution and Licensing](#attribution-and-licensing)

## ✨ Features

- **Automated Data Download**: Automatically downloads the arXiv metadata snapshot from Google Drive
- **Chunked Processing**: Processes large datasets in manageable chunks to optimize memory usage
- **High-Performance Embeddings**: Uses `all-roberta-large-v1` Sentence Transformer model for state-of-the-art semantic embeddings
- **L2 Normalization**: Automatically normalizes embeddings for efficient cosine similarity calculations
- **Fault Tolerant**: Processes data in chunks, allowing for recovery from interruptions
- **Efficient Storage**: Saves results in HDF5 format for fast I/O and compression
- **Automatic Cleanup**: Removes temporary chunk files after merging

## 📦 Prerequisites

- **Python**: 3.8 or higher
- **Jupyter Notebook** or **JupyterLab**
- **Git** (for cloning the EmbedX repository)
- **Internet Connection** (for downloading metadata and models)
- **Sufficient Disk Space**: 
  - arXiv metadata snapshot: ~2.3 GB
  - Embeddings: ~1-2 MB per 1000 papers (depending on model dimension)
- **RAM**: Recommended 8GB+ (16GB+ for processing large batches)

## 🚀 Installation

### Step 1: Clone or Navigate to the Repository

```bash
cd paper_embedder
```

### Step 2: Install Dependencies

The notebook will automatically install all required dependencies. However, you can also install them manually:

```bash
# Clone EmbedX repository
git clone https://github.com/huynguyen6906/EmbedX.git

# Install EmbedX dependencies
pip install -r EmbedX/requirements.txt

# Install EmbedX package
cd EmbedX && pip install . && cd ..

# Install additional dependency
pip install gdown
```

### Required Packages

- `torch` - PyTorch for deep learning
- `numpy` - Numerical computing
- `h5py` - HDF5 file handling
- `tqdm` - Progress bars
- `sentence-transformers` - Sentence embedding models
- `gdown` - Google Drive downloader
- `embedx` - Custom embedding utilities

## ⚙️ Configuration

Before running the notebook, configure the processing parameters in **Cell 5**:

```python
start = 0        # Starting index (0-based)
end = 100        # Ending index (exclusive)
chunk = 10       # Number of papers to process per chunk
file_path = '.cache/arxiv-metadata-oai-snapshot.json'
model = SentenceTransformer('all-roberta-large-v1')
```

### Configuration Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `start` | First paper index to process (0-based) | `0` |
| `end` | Last paper index to process (exclusive) | `1000` |
| `chunk` | Number of papers processed per chunk | `10`, `100`, `1000` |
| `file_path` | Path to arXiv metadata file | `'.cache/arxiv-metadata-oai-snapshot.json'` |
| `model` | Sentence Transformer model name | `'all-roberta-large-v1'` |

### Recommended Chunk Sizes

- **Small datasets (< 10K papers)**: `chunk = 100`
- **Medium datasets (10K - 100K papers)**: `chunk = 1000`
- **Large datasets (> 100K papers)**: `chunk = 5000`

**Note**: Smaller chunks use less memory but have more overhead. Adjust based on your available RAM.

## 📖 Usage

### Step-by-Step Execution

1. **Open the Notebook**
   ```bash
   jupyter notebook paper_embedder.ipynb
   ```

2. **Run Cells Sequentially**
   - **Cell 1**: Initialization and setup (clones EmbedX, installs dependencies)
   - **Cell 2**: Import required libraries
   - **Cell 3**: Download arXiv metadata (if not already cached)
   - **Cell 4**: Define helper function `merge_HDF5_files`
   - **Cell 5**: Configure processing parameters
   - **Cell 6**: Main processing loop (embeds papers in chunks)
   - **Cell 7**: Merge chunk files and cleanup

3. **Monitor Progress**
   - Progress bars show embedding progress for each chunk
   - Merge progress is displayed when combining files

### Example: Processing First 100 Papers

```python
start = 0
end = 100
chunk = 10
```

This will:
- Process 10 papers per chunk (10 chunks total)
- Create temporary files: `OUTPUT/Papers_Embedded_0.h5` through `OUTPUT/Papers_Embedded_9.h5`
- Merge into final file: `OUTPUT/Papers_Embedded_0-100.h5`
- Delete temporary chunk files

## 🔄 Workflow

### 1. Setup & Initialization
- Clones the `EmbedX` repository
- Installs all required Python packages
- Sets up the working environment

### 2. Data Preparation
- Creates `.cache` directory if it doesn't exist
- Downloads `arxiv-metadata-oai-snapshot.json` from Google Drive (if not present)
- File size: ~2.3 GB

### 3. Model Loading
- Loads the `all-roberta-large-v1` Sentence Transformer model
- Model is cached locally after first download
- Embedding dimension: 1024

### 4. Main Processing Loop (Chunking)
For each chunk:

1. **Metadata Reading**: Reads JSONL lines from the snapshot file
2. **Data Preparation**: 
   - Extracts paper IDs and abstracts
   - Generates arXiv PDF URLs: `https://arxiv.org/pdf/{paper_id}.pdf`
3. **Embedding Generation**: 
   - Encodes abstracts using the Sentence Transformer model
   - Batch size: 32 (configurable)
4. **Normalization**: 
   - Applies L2 normalization to all embeddings
   - Enables efficient cosine similarity calculations
5. **Saving**: 
   - Writes URLs and embeddings to temporary HDF5 chunk file
   - Format: `OUTPUT/Papers_Embedded_{chunk_index}.h5`
6. **Memory Cleanup**: Frees memory before next chunk

### 5. Final Aggregation
- Merges all chunk files using `merge_HDF5_files` function
- Creates unified output: `OUTPUT/Papers_Embedded_{start}-{end}.h5`
- Deletes temporary chunk files to free disk space

## 📊 Output Format

The final HDF5 file contains two datasets:

### Dataset Structure

```python
import h5py

with h5py.File('OUTPUT/Papers_Embedded_0-100.h5', 'r') as f:
    urls = f['urls'][:]          # Array of arXiv PDF URLs (bytes)
    embeddings = f['embeddings'][:] # Array of normalized embeddings (float32)
    
    print(f"Number of papers: {len(urls)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"URL dtype: {urls.dtype}")
    print(f"Embedding dtype: {embeddings.dtype}")
```

### Output Details

| Dataset | Type | Shape | Description |
|---------|------|-------|-------------|
| `urls` | `bytes` | `(N,)` | arXiv PDF URLs for each paper |
| `embeddings` | `float32` | `(N, 1024)` | L2-normalized embeddings |

### Example: Loading and Using Output

```python
import h5py
import numpy as np

# Load the embeddings
with h5py.File('OUTPUT/Papers_Embedded_0-100.h5', 'r') as f:
    urls = [url.decode('utf-8') for url in f['urls'][:]]
    embeddings = f['embeddings'][:]

# Example: Find most similar papers
query_embedding = embeddings[0]  # Use first paper as query
similarities = np.dot(embeddings, query_embedding)  # Cosine similarity (L2-normalized)
top_indices = np.argsort(similarities)[::-1][:5]  # Top 5 most similar

for idx in top_indices:
    print(f"Similarity: {similarities[idx]:.4f}, URL: {urls[idx]}")
```

## 🤖 Model Information

### Model: `all-roberta-large-v1`

- **Architecture**: RoBERTa-based Sentence Transformer
- **Embedding Dimension**: 1024
- **Performance**: State-of-the-art on semantic similarity tasks
- **License**: MIT / Apache 2.0 (check Hugging Face model card)
- **Source**: [Sentence Transformers](https://www.sbert.net/)

### Model Characteristics

- Pre-trained on large-scale datasets
- Optimized for semantic similarity and information retrieval
- Produces dense vector embeddings suitable for:
  - Semantic search
  - Paper recommendations
  - Clustering and classification
  - Similarity matching

## 🐛 Troubleshooting

### Issue: Out of Memory Errors

**Solution**: 
- Reduce `chunk` size (e.g., from 1000 to 100)
- Reduce batch size in `model.encode()` (default: 32)
- Close other applications to free RAM

### Issue: Download Fails or is Slow

**Solution**:
- Check internet connection
- The metadata file is ~2.3 GB, allow sufficient time
- File is cached in `.cache/` directory after first download

### Issue: EmbedX Repository Clone Fails

**Solution**:
```bash
# Manually clone and install
git clone https://github.com/huynguyen6906/EmbedX.git
cd EmbedX
pip install -r requirements.txt
pip install .
```

### Issue: Model Download Fails

**Solution**:
- Check Hugging Face access
- Model downloads automatically on first use
- Check available disk space
- Verify internet connection

### Issue: HDF5 File Corruption

**Solution**:
- Re-run the notebook from the failed chunk
- Delete corrupted chunk files manually
- Adjust `start` parameter to skip processed chunks

### Issue: JSON Decode Errors

**Solution**:
- The notebook automatically skips malformed JSON lines
- Check if metadata file is complete
- Re-download metadata file if necessary

## 📁 File Structure

```
paper_embedder/
├── paper_embedder.ipynb    # Main notebook
├── README.md                # This file
├── .cache/                  # Cached metadata (created automatically)
│   └── arxiv-metadata-oai-snapshot.json
├── OUTPUT/                  # Output directory (created automatically)
│   └── Papers_Embedded_0-100.h5
└── EmbedX/                  # Cloned repository (created automatically)
    └── ...
```

## ⚖️ Attribution and Licensing

### Project Code
The code in this repository (specifically `paper_embedder.ipynb` and the `merge_HDF5_files` function) is typically licensed under a standard open-source license such as the **MIT License**.

### Third-Party Assets

This project relies on assets from other sources, which are subject to their original licenses:

- **arXiv Metadata**: The `arxiv-metadata-oai-snapshot.json` file is a public domain work provided by arXiv and Cornell University, typically covered under the **Creative Commons Zero (CC0) license** for public domain dedication.

- **Sentence Transformer Model**: The `all-roberta-large-v1` model and its weights are derived from the RoBERTa architecture, often released under licenses like **MIT** or **Apache 2.0** by the original authors (e.g., Hugging Face/SBERT communities).

- **EmbedX**: The EmbedX repository and its dependencies are subject to their respective licenses as specified in their repository.

## 📝 Notes

- The first run will take longer due to model and data downloads
- Subsequent runs use cached files for faster processing
- Processing time depends on hardware (GPU recommended for large datasets)
- Embeddings are deterministic (same input = same output)
- Chunk processing allows resuming from interruptions

## 🔗 Related Resources

- [arXiv API Documentation](https://arxiv.org/help/api)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [HDF5 Documentation](https://www.h5py.org/)
- [EmbedX Repository](https://github.com/huynguyen6906/EmbedX)

---

**Last Updated**: 2025