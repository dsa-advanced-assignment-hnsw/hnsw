from setuptools import setup, find_packages

setup(
    name="embedx", 
    version="0.1.0", 
    description="A Python library to convert text and images into vector embeddings.",
    long_description=open("README.md", encoding="utf-8").read(),
    author="Nguyen Dinh Huy",
    author_email="dinhhuy6906@gmail.com",
    url="https://github.com/dsa-advanced-assignment-hnsw/embedx.git",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "h5py",
        "tqdm",
        "PyMuPDF",
        "sentence-transformers",
        "Pillow",
        "ftfy",
        "regex",
        "clip-anytorch",
        "python-docx",
        "bs4",
        "scikit-learn"
    ],
    python_requires=">=3.7",
    classifiers=[  
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)