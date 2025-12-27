import sys
import os
import numpy as np
from typing import List, Optional, Any, Dict, Union

# Add project root to sys.path to allow importing simple_hnsw
# Assuming this script is in backend/, and simple_hnsw is in root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

try:
    from simple_hnsw.src.simple_hnsw.hnsw import HNSW
except ImportError as e:
    print(f"Error importing HNSW: {e}")
    # Try another common path structure
    try:
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../simple_hnsw/src')))
        from simple_hnsw.hnsw import HNSW
    except ImportError:
        print("Final attempt to import HNSW failed.")
        HNSW = None

from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="HNSW Visualization Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
hnsw_instance: Optional[HNSW] = None
# Keep track of generated data for consistency
generated_data: List[List[float]] = []

class InitParams(BaseModel):
    m: int = 16
    ef_construction: int = 200
    max_elements: int = 50
    dim: int = 2
    init_count: int = 20 

class SearchParams(BaseModel):
    vector: List[float]
    k: int = 5
    ef: Optional[int] = None

class InsertParams(BaseModel):
    vector: List[float]

def convert_to_json_safe(obj: Any) -> Any:
    """Recursively converts NumPy types, sets, and tuples to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: convert_to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_json_safe(i) for i in obj]
    if isinstance(obj, (set, tuple)):
        return [convert_to_json_safe(i) for i in obj]
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return convert_to_json_safe(obj.tolist())
    return obj

@app.get("/")
def read_root():
    return {"status": "active", "message": "HNSW Visualization Backend Running"}

@app.post("/api/init_random")
def init_random(params: InitParams):
    global hnsw_instance, generated_data
    
    if HNSW is None:
        raise HTTPException(status_code=500, detail="HNSW class could not be imported")

    print(f"Initializing HNSW with M={params.m}, ef_construction={params.ef_construction}, dim={params.dim}")
    
    # 1. Create Instance
    hnsw_instance = HNSW(space='l2', dim=params.dim)
    
    # 2. Configure Index
    hnsw_instance.init_index(
        max_elements=params.max_elements,
        M=params.m,
        ef_construction=params.ef_construction
    )
    
    # Generate random data for insertion pool
    count = min(params.init_count, params.max_elements)
    print(f"Generating {count} random vectors...")
    
    data = np.random.rand(count, params.dim).astype(np.float32)
    generated_data = data.tolist()
    
    # Pre-insert data
    for vec in data:
         hnsw_instance.insert(vec)
    
    return {
        "message": "Initialized",
        "vectors": generated_data,
        "config": params.model_dump() # Pydantic v2
    }

@app.post("/api/insert")
def insert_node(params: InsertParams):
    global hnsw_instance
    if not hnsw_instance:
        raise HTTPException(status_code=400, detail="HNSW not initialized. Call /api/init_random first.")
    
    vector_arr = np.array(params.vector, dtype=np.float32)
    logs = []

    def logger(param):
        logs.append(param)
        
    current_id = hnsw_instance.cur_element_count
    hnsw_instance.insert(vector_arr, logger=logger)
    
    node_id = current_id

    return {
        "node_id": int(node_id),
        "logs": convert_to_json_safe(logs)
    }

@app.post("/api/search")
def search_node(params: SearchParams):
    global hnsw_instance
    if not hnsw_instance:
        raise HTTPException(status_code=400, detail="HNSW not initialized")

    vector_arr = np.array(params.vector, dtype=np.float32)
    logs = []

    def logger(param):
        logs.append(param)
        
    try:
        # Dynamically set ef if provided
        if params.ef is not None:
             hnsw_instance.ef = params.ef

        results = hnsw_instance.knn_search(vector_arr, K=params.k, logger=logger)
    except Exception as e:
        print(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    return {
        "results": [int(i) for i in results],
        "logs": convert_to_json_safe(logs)
    }

@app.get("/api/state")
def get_state():
    global hnsw_instance
    if not hnsw_instance:
        return {"nodes": [], "edges": [], "global_entry_point": None}
    
    layout = hnsw_instance._compute_layout()
    
    nodes = []
    edges = []
    
    for (nid, layer), pos in layout.items():
        nodes.append({
            "id": nid,
            "layer": layer,
            "x": pos[0], "y": pos[1], "z": pos[2],
            "vector": convert_to_json_safe(hnsw_instance.data[nid])
        })
        
    for l in range(len(hnsw_instance.graph)):
        for u, neighbors in hnsw_instance.graph[l].items():
            for v in neighbors:
                if u < v:
                     edges.append({"u": u, "v": v, "layer": l})
                     
    return {
        "nodes": nodes, 
        "edges": edges,
        "global_entry_point": hnsw_instance.entry_point
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    print(f"Starting HNSW Visualization Server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
