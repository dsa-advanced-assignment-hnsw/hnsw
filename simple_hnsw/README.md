# Simple-HNSW

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A pure Python implementation of the **Hierarchical Navigable Small World (HNSW)** algorithm for approximate nearest neighbor search. This project is designed primarily for **educational purposes**, offering a clean, readable codebase to understand the inner workings of HNSW.

It now features a powerful **Modern Web Application** for interactive 3D visualization of the graph construction and search process.

## ✨ Features

- **Pure Python Implementation**: Easy to read, debug, and modify. Ideal for learning the HNSW algorithm.
- **Distance Metrics**: Supports **L2 (Euclidean)** and **Cosine** distance metrics.
- **Interactive 3D Web Visualization** (New!):
    - **React + Three.js**: High-performance 3D rendering of the HNSW graph.
    - **Real-time Animation**: Watch node insertion, layer transitions, and nearest neighbor search step-by-step.
    - **Modern UI**: Dark/Light mode, glassmorphism design, and intuitive controls.
    - **FastAPI Backend**: Robust API to serve graph state and simulation logs.
- **Benchmarking Suite**: Comprehensive tools to compare performance against `hnswlib`.

## 🚀 Quick Start

The easiest way to run the visualization is using the provided helper scripts.

### Prerequisites
- **Python 3.8+**
- **Node.js 16+** & **npm**

### Step 1: Clone the repository
```bash
git clone https://github.com/dsa-advanced-assignment-hnsw/Simple-HNSW.git
cd Simple-HNSW
```

### Step 2: Run the Application

#### 🐧 Linux / macOS
Run the shell script:
```bash
./run_web_app.sh
```
*Note: This script will automatically create the virtual environment, install dependencies, and start both the backend and frontend.*

#### 🪟 Windows
Run the batch file (double-click or run in CMD):
```cmd
run_web_app.bat
```
*Note: This script will automatically create the virtual environment, install dependencies, and launch two command windows (one for Backend, one for Frontend).*

Open **http://localhost:5173** in your browser to start exploring!

## 🛠️ Manual Installation & Running

If you prefer to run services manually or debug specific parts:

### 1. Backend (FastAPI)

**Linux / macOS:**
```bash
# Create venv (first time)
python3 -m venv .venv

# Activate & Install
source .venv/bin/activate
pip install -r web_app/backend/requirements.txt

# Run Server
python web_app/backend/server.py
```

**Windows:**
```cmd
:: Create venv (first time)
python -m venv .venv

:: Activate & Install
.venv\Scripts\activate
pip install -r web_app/backend/requirements.txt

:: Run Server
python web_app/backend/server.py
```
Server runs on `http://localhost:8000`.

### 2. Frontend (React/Vite)

```bash
# From project root
cd web_app/frontend
npm install  # Install dependencies
npm run dev  # Start dev server
```
App runs on `http://localhost:5173`.

## 📖 Python Library Usage

You can still use the core HNSW library directly in Python scripts.

```python
import numpy as np
from src.simple_hnsw.hnsw import HNSW

# 1. Generate random data
dim = 100
num_elements = 1000
data = np.random.rand(num_elements, dim)

# 2. Initialize HNSW index
index = HNSW(space='l2', dim=dim)
index.init_index(max_elements=num_elements, M=16, ef_construction=200)

# 3. Insert items
index.insert_items(data)

# 4. Search
query = np.random.rand(dim)
neighbors = index.knn_search(query, k=10)

print("Nearest Neighbors Indices:", neighbors)
```

## ⚡ Benchmarking

Compare performance against `hnswlib` (requires `hnswlib` installed):

```bash
python tests/benchmark_hnsw.py
```

## 📂 Project Structure

```
Simple-HNSW/
├── src/
│   └── simple_hnsw/         # Core HNSW Python implementation
├── web_app/
│   ├── backend/             # FastAPI Server & API
│   └── frontend/            # React + Three.js Visualization
├── tests/                   # Benchmarks and Unit Tests
├── run_web_app.sh           # Linux/Mac Startup Script
├── run_web_app.bat          # Windows Startup Script
└── requirements.txt         # Core dependencies
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.