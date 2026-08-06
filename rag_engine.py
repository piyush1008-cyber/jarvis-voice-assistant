import re
import numpy as np
import pypdf

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

class LocalRAGEngine:
    def __init__(self):
        self.chunks = []
        self.embeddings = None
        self.model = None # Loaded lazily to prevent UI freezing on startup

    def load_model(self):
        """
        Loads the SentenceTransformer model lazily if not already loaded.
        """
        if self.model is None and EMBEDDINGS_AVAILABLE:
            try:
                # Load a lightweight, high-performance model (~80MB, fast on CPU)
                # It automatically downloads on first call and caches locally
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                return True
            except Exception as e:
                print(f"Error loading SentenceTransformer model: {e}")
                return False
        return self.model is not None

    def process_pdf(self, pdf_file_path_or_stream):
        """
        Parses a PDF document, splits text into overlapping chunks,
        and computes semantic embeddings locally.
        """
        self.chunks = []
        self.embeddings = None
        
        # Ensure model is loaded before processing
        if not self.load_model():
            print("SentenceTransformer model is not loaded. Cannot index PDF.")
            return False
            
        full_text = ""
        try:
            reader = pypdf.PdfReader(pdf_file_path_or_stream)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
        except Exception as e:
            print(f"Error parsing PDF: {e}")
            return False

        if not full_text.strip():
            print("PDF extracted text is empty.")
            return False

        # Normalize whitespace (replace multiple spaces/newlines with a single space)
        normalized_text = re.sub(r'\s+', ' ', full_text)

        # Chunk text (~500 characters per chunk, with 100 character overlap)
        chunk_size = 500
        overlap = 100
        
        i = 0
        while i < len(normalized_text):
            chunk = normalized_text[i : i + chunk_size]
            if len(chunk.strip()) > 30: # Only store meaningful chunks
                self.chunks.append(chunk.strip())
            i += (chunk_size - overlap)

        if not self.chunks:
            return False

        # Generate local vector embeddings
        try:
            # Convert list of text chunks to numpy array of vectors
            self.embeddings = self.model.encode(self.chunks, convert_to_numpy=True)
            return True
        except Exception as e:
            print(f"Error computing embeddings: {e}")
            return False

    def search(self, query, top_k=3):
        """
        Performs vector search using Cosine Similarity on numpy arrays.
        Returns:
            list of dicts containing 'text' and 'score' of matching chunks.
        """
        # Ensure model is loaded before searching
        if not self.load_model():
            return []
            
        if not self.chunks or self.embeddings is None:
            return []

        try:
            # Get semantic vector for query
            query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
            
            # Calculate cosine similarity: dot(A, B) / (norm(A) * norm(B))
            dot_products = np.dot(self.embeddings, query_embedding)
            norms_embeddings = np.linalg.norm(self.embeddings, axis=1)
            norm_query = np.linalg.norm(query_embedding)
            
            # Avoid division by zero
            norms_embeddings[norms_embeddings == 0] = 1e-10
            if norm_query == 0:
                norm_query = 1e-10
                
            similarities = dot_products / (norms_embeddings * norm_query)
            
            # Sort scores descending and take top_k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                results.append({
                    "text": self.chunks[idx],
                    "score": float(similarities[idx])
                })
            return results
        except Exception as e:
            print(f"Error during RAG semantic search: {e}")
            return []
