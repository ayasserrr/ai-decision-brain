from typing import List, Dict
import sys
from pathlib import Path

try:
    from configs import Config
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from configs import Config


class TextChunker:
    """Splits documents into smaller chunks with overlap"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        text = self.clean_text(text)
        
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # If this is not the last chunk, try to break at a sentence or word boundary
            if end < len(text):
                # Look for sentence endings (., !, ?)
                last_period = text.rfind('.', start, end)
                last_exclamation = text.rfind('!', start, end)
                last_question = text.rfind('?', start, end)
                
                sentence_end = max(last_period, last_exclamation, last_question)
                
                if sentence_end != -1 and sentence_end > start:
                    end = sentence_end + 1
                else:
                    # If no sentence boundary, try to break at a space
                    last_space = text.rfind(' ', start, end)
                    if last_space != -1 and last_space > start:
                        end = last_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            prev_start = start
            start = end - self.chunk_overlap
            
            # Prevent infinite loop
            if start <= prev_start:
                start = end
        
        return chunks
    
    def chunk_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, any]]:
        """Chunk all documents and return chunks with metadata"""
        all_chunks = []
        
        print(f"Chunking documents with size={self.chunk_size}, overlap={self.chunk_overlap}")
        print("-" * 50)
        
        for doc_idx, document in enumerate(documents):
            print(f"Chunking: {document['filename']}...", end=" ")
            
            chunks = self.split_text_into_chunks(document['content'])
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_data = {
                    'text': chunk,
                    'doc_id': doc_idx,
                    'chunk_id': chunk_idx,
                    'filename': document['filename'],
                    'filepath': document['filepath'],
                    'file_type': document['file_type'],
                    'total_chunks': len(chunks)
                }
                all_chunks.append(chunk_data)
            
            print(f"SUCCESS ({len(chunks)} chunks)")
        
        print("-" * 50)
        print(f"Total chunks created: {len(all_chunks)}")
        
        return all_chunks
    
    def get_chunk_stats(self, chunks: List[Dict[str, any]]) -> Dict:
        """Get statistics about chunks"""
        if not chunks:
            return {}
        
        chunk_lengths = [len(chunk['text']) for chunk in chunks]
        documents = set(chunk['filename'] for chunk in chunks)
        
        return {
            'total_chunks': len(chunks),
            'total_documents': len(documents),
            'avg_chunk_length': sum(chunk_lengths) // len(chunk_lengths),
            'min_chunk_length': min(chunk_lengths),
            'max_chunk_length': max(chunk_lengths),
            'total_characters': sum(chunk_lengths)
        }
    
    def display_stats(self, chunks: List[Dict[str, any]]):
        """Display statistics about chunks"""
        stats = self.get_chunk_stats(chunks)
        
        print("\n" + "=" * 50)
        print("Chunking Statistics")
        print("=" * 50)
        print(f"Total Chunks: {stats['total_chunks']}")
        print(f"Total Documents: {stats['total_documents']}")
        print(f"Average Chunk Length: {stats['avg_chunk_length']} characters")
        print(f"Min Chunk Length: {stats['min_chunk_length']} characters")
        print(f"Max Chunk Length: {stats['max_chunk_length']} characters")
        print(f"Total Characters: {stats['total_characters']:,}")
        print(f"Chunks per Document (avg): {stats['total_chunks'] / stats['total_documents']:.1f}")
        print("=" * 50)
    
    def display_sample_chunks(self, chunks: List[Dict[str, any]], n: int = 3):
        """Display sample chunks for inspection"""
        print("\n" + "=" * 50)
        print(f"Sample Chunks (showing first {min(n, len(chunks))})")
        print("=" * 50)
        
        for i, chunk in enumerate(chunks[:n]):
            print(f"\nChunk {i+1}:")
            print(f"  Document: {chunk['filename']}")
            print(f"  Chunk ID: {chunk['chunk_id']} / {chunk['total_chunks']}")
            print(f"  Length: {len(chunk['text'])} characters")
            print(f"  Preview: {chunk['text'][:200]}...")
            print("-" * 50)


# Test function
if __name__ == "__main__":
    from document_loader import DocumentLoader
    
    # Load documents
    loader = DocumentLoader()
    documents = loader.load_all_documents()
    
    # Chunk documents
    chunker = TextChunker()
    chunks = chunker.chunk_documents(documents)
    
    # Display statistics
    chunker.display_stats(chunks)
    chunker.display_sample_chunks(chunks, n=2)