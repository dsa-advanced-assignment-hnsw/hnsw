# Image Embedding Pipeline (CLIP & Open Images V7)

This repository contains a Jupyter Notebook (`image_embedder.ipynb`) that implements a complete, fault-tolerant pipeline for generating image embeddings.

The pipeline downloads image metadata from the Open Images V7 dataset, fetches the corresponding images, processes them using OpenAI's **CLIP (ViT-B/32)** model, and stores the resulting URL-vector pairs in HDF5 files.

##  workflow-diagram How It Works

The notebook is designed as a fault-tolerant pipeline that operates in "chunks" to handle very large datasets without running out of memory.

1.  **Setup & Tooling:**
    * Installs all required Python libraries.
    * Downloads the `downloader.py` utility from the Open Images toolkit.
    * Downloads the image ID and URL metadata files (originally from Open Images V7).

2.  **Model Loading:**
    * Loads the OpenAI **CLIP (ViT-B/32)** model and its preprocessor onto the GPU (if available).

3.  **Main Processing Loop:**
    * The script iterates through the entire dataset in chunks (e.g., 10,000 images at a time).
    * For each chunk, it:
        1.  **Downloads** the images using the `downloader.py` script.
        2.  **Preprocesses (CPU)** the images in parallel using `ThreadPoolExecutor`.
        3.  **Encodes (GPU)** the images in large batches (e.g., 512) to get the vectors.
        4.  **Saves** the results (`urls`, `embeddings`) to a temporary chunk file (e.g., `OUTPUT/Images_Embedded_0.h5`).
        5.  **Cleans up** the downloaded images to save space.

4.  **Final Aggregation:**
    * After all chunks are processed, the `merge_HDF5_files` function is called.
    * This function combines all temporary chunk files into one large, final file: `OUTPUT/Images_Embedded.h5`.
    * The temporary chunk files are then deleted.

---

## ⚖️ Attribution and Licensing

### Project Code
The code in this repository (specifically `image_embedder.ipynb` and `merge_HDF5_files` function) is licensed under the MIT License.

### Third-Party Assets
This project utilizes assets from other sources. These assets are subject to their original licenses:

* **Open Images V7:** The image metadata (IDs and URLs) and the `downloader.py` script are components of the Open Images V7 dataset. This data is licensed under the **CC BY 4.0 license**.
* **OpenAI CLIP:** The CLIP model and its pre-trained weights are (c) OpenAI and licensed under the **MIT License**.