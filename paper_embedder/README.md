# PAPER EMBEDDER: arXiv Paper Embedding Generator

This repository contains a Jupyter Notebook (`paper_embedder.ipynb`) that implements a complete pipeline for generating embeddings for scientific papers.

This process automates the downloading of the **full arXiv archive metadata** file, generates high-density vector embeddings for the paper abstracts using a **high-performance Sentence Transformer** model, and stores the results (`URL`, `Vector Embedding`) into HDF5 files.

## ⚙️ Workflow Diagram (Chunking and Fault Tolerance)

The notebook is designed as a fault-tolerant, large-data processing pipeline that operates in "chunks" to ensure memory is managed efficiently.

1.  **Setup & Initialization:**
    * Clones the repository (`EmbedX`), installs necessary Python libraries (`sentence-transformers`, `h5py`, `gdown`, etc.).
    * Downloads the **`arxiv-metadata-oai-snapshot.json`** metadata file from Google Drive if it does not exist locally.

2.  **Model Loading:**
    * Loads the **`all-roberta-large-v1`** Sentence Transformer model into memory, ready for the encoding process.

3.  **Main Processing Loop (Chunking):**
    * The script iterates through the entire dataset (from `start` to `end`) in blocks of size `chunk` (e.g., 1000 papers at a time).
    * For each chunk, it performs:
        1.  **Metadata Reading:** Reads a block of metadata (in JSONL format) from the snapshot file.
        2.  **Data Preparation:** Creates the full PDF URL and extracts the abstract content.
        3.  **Encoding:** Uses the `all-roberta-large-v1` model to convert the abstracts into embeddings.
        4.  **Normalization:** Applies **L2-normalization** to the vector embeddings to optimize cosine similarity calculation.
        5.  **Saving:** Saves the results (`urls`, `embeddings`) to a temporary HDF5 chunk file (e.g., `OUTPUT/Papers_Embedded_0.h5`).
        6.  **Cleanup:** Frees up memory used by the paper list before proceeding to the next chunk.

4.  **Final Aggregation (Merging and Cleanup):**
    * After all chunks have been processed, the custom `merge_HDF5_files` function is called.
    * This function is responsible for **merging** all temporary HDF5 chunk files into one final, unified file: `OUTPUT/Papers_Embedded_<start>-<end>.h5`.
    * Finally, all temporary HDF5 chunk files are deleted to free up disk space.

---

## ⚖️ Attribution and Licensing

### Project Code
The code in this repository (specifically `paper_embedder.ipynb` and the `merge_HDF5_files` function) is typically licensed under a standard open-source license such as the **MIT License**.

### Third-Party Assets
This project relies on assets from other sources, which are subject to their original licenses:

* **arXiv Metadata:** The `arxiv-metadata-oai-snapshot.json` file is a public domain work provided by arXiv and Cornell University, typically covered under the **Creative Commons Zero (CC0) license** for public domain dedication.
* **Sentence Transformer Model:** The `all-roberta-large-v1` model and its weights are derived from the RoBERTa architecture, often released under licenses like **MIT** or **Apache 2.0** by the original authors (e.g., Hugging Face/SBERT communities).