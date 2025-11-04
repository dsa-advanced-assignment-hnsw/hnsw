# EmbedX
EmbedX is a Python library for generating image and text embeddings using CLIP, SentenceTransformers, and other supported models.

**Notice:**  
1. Supported file types:  
   - PDF documents (`.pdf`,`.txt`,`.html`,`.docx`)  
   - RGB image files (`.png`, `.jpg`, `.jpeg`)  
2. The library **cannot** read files in subfolders of `dataset_path`.

## Installation

### Clone repo
 
```bash
git clone https://github.com/username/embedx.git
cd embedx
```

### Install required libraries

```bash
pip install -r requirements.txt
```

### Install library

*If you want to upgrade this library in the future:*

```bash
pip install -e .
```

*If you only want to use it:*

```bash
pip install .
```
## How to use

### Text methods

*Convert a text to vector (output: np.array):*

```python
embedx.text.embed_Text("The text  you want to convert")
```

```python
embedx.image.embed_Text("The text  you want to convert")
```

*Convert all text files in dataset folder (output: .h5 file saving all vector):*
```python
convert = embedx.text.EmbedX()                                  # Create a object to convert
convert.select_Dataset_Path("Dataset path. Ex: ./Dataset")
convert.select_Output_Path("Output file path. Ex: ./text.h5")
convert.embed_Dataset()                                         # Convert all files in dataset to output file
convert.embed_Other_file("file path")                           # Convert file in other directory and add the vector to the end of output file
```

### Image methods

*Convert a image to vector (output: np.array):*

```python
embedx.image.embed_Image("Image path")
```

*Convert all image files in dataset folder (output: .h5 file saving all vector):*
```python
convert = embedx.image.EmbedX()                                 # Create a object to convert
convert.select_Dataset_Path("Dataset path. Ex: ./Dataset")
convert.select_Output_Path("Output file path. Ex: ./image.h5")
convert.embed_Dataset()                                         # Convert all files in dataset to output file
convert.embed_Other_file("file path")                           # Convert file in other directory and add the vector to the end of output file
```

### Output `.h5` file

The output file is an HDF5 (`.h5`) file that contains two datasets:

- **`path`**: a string dataset storing the file paths of the files.  
- **`embeddings`**: a NumPy array dataset storing the corresponding embedding vectors for each file.  

Each entry in `path` corresponds to the embedding vector at the same index in `embeddings`.

More Information About .h5 Files: [here](https://www.neonscience.org/resources/learning-hub/tutorials/about-hdf5).