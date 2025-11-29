import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import sys
from pathlib import Path

try:
    from configs import Config
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from configs import Config


class EmbeddingGenerator:
    """Generates embeddings for text chunks using sentence transformers"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or Config.EMBEDDING_MODEL_NAME
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print(f"Model loaded successfully")
        print(f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """Generate embeddings for multiple texts in batches"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings
    
    def embed_chunks(self, chunks: List[Dict[str, any]], batch_size: int = 32) -> List[Dict[str, any]]:
        """Generate embeddings for all chunks"""
        print(f"\nGenerating embeddings for {len(chunks)} chunks")
        print("-" * 50)
        
        # Extract texts from chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings in batches
        embeddings = self.generate_embeddings_batch(texts, batch_size=batch_size)
        
        # Add embeddings to chunks
        embedded_chunks = []
        for i, chunk in enumerate(chunks):
            embedded_chunk = chunk.copy()
            embedded_chunk['embedding'] = embeddings[i]
            embedded_chunks.append(embedded_chunk)
        
        print("-" * 50)
        print(f"Successfully generated {len(embedded_chunks)} embeddings")
        print(f"Embedding shape: {embeddings[0].shape}")
        
        return embedded_chunks
    
    def get_embedding_stats(self, embedded_chunks: List[Dict[str, any]]) -> Dict:
        """Get statistics about embeddings"""
        if not embedded_chunks:
            return {}
        
        embeddings = np.array([chunk['embedding'] for chunk in embedded_chunks])
        
        return {
            'total_embeddings': len(embedded_chunks),
            'embedding_dimension': embeddings.shape[1],
            'mean_norm': float(np.mean(np.linalg.norm(embeddings, axis=1))),
            'std_norm': float(np.std(np.linalg.norm(embeddings, axis=1))),
            'total_size_mb': float(embeddings.nbytes / (1024 * 1024))
        }
    
    def display_stats(self, embedded_chunks: List[Dict[str, any]]):
        """Display statistics about embeddings"""
        stats = self.get_embedding_stats(embedded_chunks)
        
        print("\n" + "=" * 50)
        print("Embedding Statistics")
        print("=" * 50)
        print(f"Total Embeddings: {stats['total_embeddings']}")
        print(f"Embedding Dimension: {stats['embedding_dimension']}")
        print(f"Mean Embedding Norm: {stats['mean_norm']:.4f}")
        print(f"Std Embedding Norm: {stats['std_norm']:.4f}")
        print(f"Total Size: {stats['total_size_mb']:.2f} MB")
        print("=" * 50)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def find_similar_chunks(self, query: str, embedded_chunks: List[Dict[str, any]], top_k: int = 5) -> List[tuple]:
        """Find most similar chunks to a query"""
        # Generate embedding for query
        query_embedding = self.generate_embedding(query)
        
        # Compute similarities
        similarities = []
        for i, chunk in enumerate(embedded_chunks):
            similarity = self.compute_similarity(query_embedding, chunk['embedding'])
            similarities.append((i, similarity, chunk))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


# Test function
if __name__ == "__main__":
    from document_loader import DocumentLoader
    from text_chunker import TextChunker
    
    # Load and chunk documents
    print("Step 1: Loading documents...")
    loader = DocumentLoader()
    documents = loader.load_all_documents()
    
    print("\nStep 2: Chunking documents...")
    chunker = TextChunker()
    chunks = chunker.chunk_documents(documents)
    
    print("\nStep 3: Generating embeddings...")
    embedder = EmbeddingGenerator()
    embedded_chunks = embedder.embed_chunks(chunks)
    
    # Display statistics
    embedder.display_stats(embedded_chunks)
    
    # Test similarity search
    print("\n" + "=" * 50)
    print("Testing Similarity Search")
    print("=" * 50)
    test_query = "What is the financial performance?"
    print(f"Query: '{test_query}'")
    print("-" * 50)
    
    similar_chunks = embedder.find_similar_chunks(test_query, embedded_chunks, top_k=3)
    
    for rank, (idx, similarity, chunk) in enumerate(similar_chunks, 1):
        print(f"\nRank {rank} (Similarity: {similarity:.4f}):")
        print(f"  Document: {chunk['filename']}")
        print(f"  Chunk: {chunk['chunk_id']}/{chunk['total_chunks']}")
        print(f"  Text preview: {chunk['text'][:150]}...")
    
    print("\n" + "=" * 50)