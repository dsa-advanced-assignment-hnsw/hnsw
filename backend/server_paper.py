import os
import numpy as np
import h5py
import hnswlib
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import PyPDF2
import io

app = Flask(__name__)
CORS(app, origins=['*'])
H5_FILE_PATH='Papers_Embedbed_0-100000.h5'
os.environ['H5_FILE_PATH'] = H5_FILE_PATH

class PaperSearchEngine:
    def __init__(self, h5_file_path=H5_FILE_PATH):
        """
        Initialize the Paper Search Engine with Sentence Transformers and HNSW index.

        Args:
            h5_file_path: Path to the HDF5 file containing paper embeddings and URLs
        """
        print("Initializing Paper Search Engine...")

        # Load Sentence Transformer model (same model used for embeddings)
        print("Loading Sentence Transformer model (all-roberta-large-v1)...")
        self.model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
        print("Model loaded successfully!")

        # Determine the .bin filename
        index_filename = os.path.splitext(h5_file_path)[0] + ".bin"

        # Load embeddings and URLs from HDF5
        print(f"Loading embeddings from {h5_file_path}...")
        with h5py.File(h5_file_path, 'r') as f:
            self.embeddings = f['embeddings'][:]
            # Decode URLs from bytes to strings
            self.urls = [url.decode('utf-8') if isinstance(url, bytes) else url
                        for url in f['urls'][:]]

        print(f"Loaded {len(self.embeddings)} paper embeddings")
        print(f"Embedding dimension: {self.embeddings.shape[1]}")

        dim = self.embeddings.shape[1]
        max_elements = len(self.embeddings)

        # Check if .bin file exists for faster loading
        if os.path.exists(index_filename):
            print(f"⚡ Found existing HNSW index: {index_filename}")
            print(f"Loading HNSW index from .bin file (fast mode)...")
            self.index = hnswlib.Index(space='cosine', dim=dim)
            self.index.load_index(index_filename, max_elements=max_elements)
            self.index.set_ef(200)
            print("✅ HNSW index loaded successfully from .bin file!")
        else:
            # Build HNSW index from scratch if .bin doesn't exist
            print("Building HNSW index from scratch...")
            print("(This may take a while for the first run)")
            self.index = hnswlib.Index(space='cosine', dim=dim)

            # Initialize index with capacity
            self.index.init_index(
                max_elements=max_elements,
                ef_construction=400,  # Higher = better quality, slower build
                M=200  # Number of connections per element
            )

            # Add embeddings to index
            self.index.add_items(self.embeddings, np.arange(len(self.embeddings)))

            # Set ef for search (higher = more accurate, slower)
            self.index.set_ef(200)

            # Save index for future runs
            self.index.save_index(index_filename)
            print(f"💾 Saved HNSW index to: {index_filename}")
            print("   (Next startup will be faster!)")

        print("HNSW index built successfully!")
        print("Paper Search Engine ready!")

    def text_to_vector(self, text):
        """
        Convert text to embedding vector using Sentence Transformer.

        Args:
            text: Input text (query, abstract, etc.)

        Returns:
            numpy array: L2-normalized embedding vector
        """
        # Encode text and normalize
        embedding = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        return embedding[0]

    def extract_text_from_file(self, file_content, file_extension):
        """
        Extract text from uploaded file.

        Args:
            file_content: File content as bytes
            file_extension: File extension (.txt, .pdf, .md)

        Returns:
            str: Extracted text
        """
        if file_extension in ['.txt', '.md']:
            # Plain text files
            try:
                return file_content.decode('utf-8')
            except UnicodeDecodeError:
                return file_content.decode('latin-1')

        elif file_extension == '.pdf':
            # PDF files
            try:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
            except Exception as e:
                raise ValueError(f"Error extracting text from PDF: {str(e)}")

        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    def search(self, query_text, k=10):
        """
        Search for similar papers using text query.

        Args:
            query_text: Text query (keywords, sentence, abstract, etc.)
            k: Number of results to return

        Returns:
            list: List of (url, similarity_score) tuples
        """
        # Convert query to vector
        query_vector = self.text_to_vector(query_text)

        # Search using HNSW
        labels, distances = self.index.knn_query(query_vector, k=k)

        # Convert cosine distance to similarity (1 - distance)
        similarities = 1 - distances[0]

        # Get results
        results = []
        for idx, similarity in zip(labels[0], similarities):
            results.append({
                'url': self.urls[idx],
                'similarity': float(similarity)
            })

        return results

    def search_by_file(self, file_content, file_extension, k=10):
        """
        Search for similar papers using uploaded file content.

        Args:
            file_content: File content as bytes
            file_extension: File extension
            k: Number of results to return

        Returns:
            list: List of (url, similarity_score) tuples
        """
        # Extract text from file
        text = self.extract_text_from_file(file_content, file_extension)

        # Use regular text search
        return self.search(text, k)


# Initialize search engine (singleton)
print("Starting Flask server...")
H5_FILE_PATH = os.getenv('H5_FILE_PATH', 'Papers_Embedbed_0-10000.h5')
search_engine = PaperSearchEngine(h5_file_path=H5_FILE_PATH)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'paper-search-engine',
        'total_papers': len(search_engine.urls),
        'embedding_dim': search_engine.embeddings.shape[1],
        'model': 'all-roberta-large-v1'
    })


@app.route('/search', methods=['POST'])
def search_text():
    """
    Text-based paper search endpoint.

    Expects JSON:
    {
        "query": "machine learning transformers",
        "k": 10
    }
    """
    try:
        data = request.get_json()

        if not data or 'query' not in data:
            return jsonify({'error': 'Missing query parameter'}), 400

        query = data['query']
        k = data.get('k', 10)

        # Validate k
        if not isinstance(k, int) or k < 1 or k > 100:
            return jsonify({'error': 'k must be an integer between 1 and 100'}), 400

        # Perform search
        results = search_engine.search(query, k=k)

        return jsonify({
            'query': query,
            'k': k,
            'results': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/search/file', methods=['POST'])
def search_file():
    """
    File-based paper search endpoint.

    Expects multipart/form-data with:
    - file: The uploaded file (.txt, .pdf, .md)
    - k: Number of results (optional, default 10)
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        # Get file extension
        file_extension = os.path.splitext(file.filename)[1].lower()

        if file_extension not in ['.txt', '.pdf', '.md']:
            return jsonify({'error': 'Unsupported file type. Supported: .txt, .pdf, .md'}), 400

        # Read file content
        file_content = file.read()

        # Get k parameter
        k = request.form.get('k', 10, type=int)

        # Validate k
        if k < 1 or k > 100:
            return jsonify({'error': 'k must be between 1 and 100'}), 400

        # Perform search
        results = search_engine.search_by_file(file_content, file_extension, k=k)

        return jsonify({
            'filename': file.filename,
            'k': k,
            'results': results
        })

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))  # Default to 5001 to avoid conflict with image server
    debug = os.getenv('FLASK_DEBUG', '0') == '1'

    print(f"\nPaper Search Engine running on http://localhost:{port}")
    print(f"Health check: http://localhost:{port}/health")
    print(f"Total papers indexed: {len(search_engine.urls)}")

    app.run(host='0.0.0.0', port=port, debug=debug)
