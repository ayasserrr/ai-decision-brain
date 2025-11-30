import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import sys
from pathlib import Path

try:
    from configs import Config
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from configs import Config


class VectorStore:
    """FAISS-based vector store for efficient similarity search"""
    
    def __init__(self, dimension: int = None):
        self.dimension = dimension or Config.EMBEDDING_DIMENSION
        self.index = None
        self.chunks_metadata = []
        self.is_trained = False
    
    def create_index(self, index_type: str = "flat"):
        """Create a FAISS index"""
        if index_type == "flat":
            # L2 distance (Euclidean)
            self.index = faiss.IndexFlatL2(self.dimension)
        elif index_type == "cosine":
            # Cosine similarity (using Inner Product with normalized vectors)
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        print(f"Created FAISS index: {index_type}")
        print(f"Index dimension: {self.dimension}")
    
    def normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return vectors / norms
    
    def add_embeddings(self, embedded_chunks: List[Dict[str, any]], normalize: bool = True):
        """Add embeddings to the FAISS index"""
        if self.index is None:
            self.create_index(index_type="cosine" if normalize else "flat")
        
        print(f"\nAdding {len(embedded_chunks)} embeddings to vector store")
        print("-" * 50)
        
        # Extract embeddings
        embeddings = np.array([chunk['embedding'] for chunk in embedded_chunks]).astype('float32')
        
        # Normalize if using cosine similarity
        if normalize:
            embeddings = self.normalize_vectors(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store metadata (without embeddings to save memory)
        for chunk in embedded_chunks:
            metadata = {k: v for k, v in chunk.items() if k != 'embedding'}
            self.chunks_metadata.append(metadata)
        
        self.is_trained = True
        
        print(f"Successfully added {self.index.ntotal} vectors to index")
        print("-" * 50)
    
    def search(self, query_embedding: np.ndarray, top_k: int = None, normalize: bool = True) -> List[Tuple[int, float, Dict]]:
        """Search for similar vectors"""
        if not self.is_trained:
            raise ValueError("Index is empty. Add embeddings first.")
        
        top_k = top_k or Config.TOP_K_RESULTS
        
        # Prepare query
        query_vector = query_embedding.reshape(1, -1).astype('float32')
        
        # Normalize if using cosine similarity
        if normalize:
            query_vector = self.normalize_vectors(query_vector)
        
        # Search
        distances, indices = self.index.search(query_vector, top_k)
        
        # Prepare results
        results = []
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            if idx != -1:  # Valid result
                # Convert distance to similarity score
                # For cosine similarity (Inner Product): higher is better, use as-is
                # For L2 distance: lower is better, so invert to make higher scores better
                similarity = float(distance) if normalize else 1.0 / (1.0 + float(distance))
                results.append((int(idx), similarity, self.chunks_metadata[idx]))
        
        return results
    
    def save(self, index_path: Path = None, metadata_path: Path = None):
        """Save the FAISS index and metadata to disk"""
        if not self.is_trained:
            raise ValueError("Cannot save an empty index")
        
        index_path = index_path or Config.VECTOR_DB_DIR / "faiss_index.bin"
        metadata_path = metadata_path or Config.VECTOR_DB_DIR / "metadata.pkl"
        
        print(f"\nSaving vector store...")
        print("-" * 50)
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        print(f"FAISS index saved to: {index_path}")
        
        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.chunks_metadata, f)
        print(f"Metadata saved to: {metadata_path}")
        
        print("-" * 50)
        print("Vector store saved successfully")
    
    def load(self, index_path: Path = None, metadata_path: Path = None):
        """Load the FAISS index and metadata from disk"""
        index_path = index_path or Config.VECTOR_DB_DIR / "faiss_index.bin"
        metadata_path = metadata_path or Config.VECTOR_DB_DIR / "metadata.pkl"
        
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        print(f"\nLoading vector store...")
        print("-" * 50)
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        print(f"FAISS index loaded from: {index_path}")
        print(f"Index contains {self.index.ntotal} vectors")
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            self.chunks_metadata = pickle.load(f)
        print(f"Metadata loaded from: {metadata_path}")
        print(f"Loaded {len(self.chunks_metadata)} metadata entries")
        
        self.is_trained = True
        self.dimension = self.index.d
        
        print("-" * 50)
        print("Vector store loaded successfully")
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        if not self.is_trained:
            return {'status': 'empty'}
        
        return {
            'status': 'ready',
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'total_metadata': len(self.chunks_metadata)
        }
    
    def display_stats(self):
        """Display statistics about the vector store"""
        stats = self.get_stats()
        
        print("\n" + "=" * 50)
        print("Vector Store Statistics")
        print("=" * 50)
        print(f"Status: {stats['status']}")
        
        if stats['status'] == 'ready':
            print(f"Total Vectors: {stats['total_vectors']}")
            print(f"Dimension: {stats['dimension']}")
            print(f"Metadata Entries: {stats['total_metadata']}")
        
        print("=" * 50)


# Test function
if __name__ == "__main__":
    from document_loader import DocumentLoader
    from text_chunker import TextChunker
    from embedding_generator import EmbeddingGenerator
    
    # Step 1: Load documents
    print("Step 1: Loading documents...")
    loader = DocumentLoader()
    documents = loader.load_all_documents()
    
    # Step 2: Chunk documents
    print("\nStep 2: Chunking documents...")
    chunker = TextChunker()
    chunks = chunker.chunk_documents(documents)
    
    # Step 3: Generate embeddings
    print("\nStep 3: Generating embeddings...")
    embedder = EmbeddingGenerator()
    embedded_chunks = embedder.embed_chunks(chunks)
    
    # Step 4: Create and populate vector store
    print("\nStep 4: Creating vector store...")
    vector_store = VectorStore()
    vector_store.add_embeddings(embedded_chunks, normalize=True)
    vector_store.display_stats()
    
    # Step 5: Save vector store
    vector_store.save()
    
    # Step 6: Test search
    print("\n" + "=" * 50)
    print("Testing Vector Store Search")
    print("=" * 50)
    test_query = "What are the marketing strategies?"
    print(f"Query: '{test_query}'")
    
    query_embedding = embedder.generate_embedding(test_query)
    results = vector_store.search(query_embedding, top_k=3, normalize=True)
    
    print("-" * 50)
    for rank, (idx, similarity, metadata) in enumerate(results, 1):
        print(f"\nRank {rank} (Similarity: {similarity:.4f}):")
        print(f"  Document: {metadata['filename']}")
        print(f"  Chunk: {metadata['chunk_id']}/{metadata['total_chunks']}")
        print(f"  Text preview: {metadata['text'][:150]}...")
    
    print("\n" + "=" * 50)
    print("SUCCESS: Vector store is ready for use!")