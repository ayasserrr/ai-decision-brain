import sys
from pathlib import Path
import openai
from typing import List, Dict, Tuple

try:
    from src import EmbeddingGenerator, VectorStore
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src import EmbeddingGenerator, VectorStore

try:
    from configs import Config
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from configs import Config


class RAGPipeline:
    """Complete RAG pipeline for question answering"""
    
    def __init__(self, vector_store_path: str = None):
        # Initialize OpenAI client
        self.client = openai.OpenAI(
            base_url=Config.OPENAI_API_URL,
            api_key=Config.OPENAI_API_KEY
        )
        
        # Initialize components
        print("Initializing RAG Pipeline...")
        print("-" * 50)
        
        self.embedder = EmbeddingGenerator()
        self.vector_store = VectorStore()
        
        # Load vector store
        print("\nLoading vector store...")
        self.vector_store.load()
        
        print("-" * 50)
        print("RAG Pipeline initialized successfully")
    
    def retrieve_context(self, query: str, top_k: int = None) -> List[Tuple[int, float, Dict]]:
        """Retrieve relevant context for a query"""
        top_k = top_k or Config.TOP_K_RESULTS
        
        # Generate query embedding
        query_embedding = self.embedder.generate_embedding(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=top_k, normalize=True)
        
        return results
    
    def format_context(self, results: List[Tuple[int, float, Dict]]) -> str:
        """Format retrieved chunks into context string"""
        context_parts = []
        
        for rank, (idx, similarity, metadata) in enumerate(results, 1):
            context_parts.append(
                f"[Document {rank}: {metadata['filename']}]\n"
                f"{metadata['text']}\n"
            )
        
        return "\n".join(context_parts)
    
    def create_prompt(self, query: str, context: str) -> str:
        """Create prompt for the LLM"""
        prompt = f"""You are a helpful AI assistant. Answer the question based on the provided context.

Context:
{context}

Question: {query}

Instructions:
- Answer the question based only on the provided context
- If the context doesn't contain enough information to answer the question, say so
- Be concise and specific
- Cite the document name when providing information

Answer:"""
        
        return prompt
    
    def generate_answer(self, query: str, context: str) -> Dict[str, any]:
        """Generate answer using the LLM"""
        prompt = self.create_prompt(query, context)
        
        try:
            response = self.client.chat.completions.create(
                model=Config.GENERATION_MODEL_ID,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=Config.MAX_TOKENS,
                temperature=Config.TEMPERATURE
            )
            
            answer = response.choices[0].message.content
            
            return {
                'answer': answer,
                'model': Config.GENERATION_MODEL_ID,
                'tokens_used': response.usage.total_tokens if hasattr(response, 'usage') else None
            }
        
        except Exception as e:
            return {
                'answer': f"Error generating answer: {str(e)}",
                'model': Config.GENERATION_MODEL_ID,
                'tokens_used': None,
                'error': str(e)
            }
    
    def query(self, question: str, top_k: int = None, return_context: bool = False) -> Dict[str, any]:
        """Complete RAG query pipeline"""
        print(f"\nQuery: {question}")
        print("-" * 50)
        
        # Retrieve context
        print("Retrieving relevant context...")
        results = self.retrieve_context(question, top_k=top_k)
        
        print(f"Found {len(results)} relevant chunks")
        
        # Format context
        context = self.format_context(results)
        
        # Generate answer
        print("Generating answer...")
        response = self.generate_answer(question, context)
        
        # Prepare output
        output = {
            'question': question,
            'answer': response['answer'],
            'sources': [
                {
                    'filename': metadata['filename'],
                    'chunk_id': metadata['chunk_id'],
                    'similarity': similarity
                }
                for idx, similarity, metadata in results
            ],
            'model': response['model'],
            'tokens_used': response.get('tokens_used')
        }
        
        if return_context:
            output['context'] = context
            output['retrieved_chunks'] = [
                {
                    'text': metadata['text'],
                    'filename': metadata['filename'],
                    'similarity': similarity
                }
                for idx, similarity, metadata in results
            ]
        
        return output
    
    def display_result(self, result: Dict[str, any], show_sources: bool = True):
        """Display query result in a formatted way"""
        print("\n" + "=" * 50)
        print("QUESTION")
        print("=" * 50)
        print(result['question'])
        
        print("\n" + "=" * 50)
        print("ANSWER")
        print("=" * 50)
        print(result['answer'])
        
        if show_sources:
            print("\n" + "=" * 50)
            print("SOURCES")
            print("=" * 50)
            for i, source in enumerate(result['sources'], 1):
                print(f"{i}. {source['filename']} (Similarity: {source['similarity']:.4f})")
        
        if result.get('tokens_used'):
            print(f"\nTokens used: {result['tokens_used']}")
        
        print("=" * 50)


# Test function
if __name__ == "__main__":
    # Initialize RAG pipeline
    rag = RAGPipeline()
    
    # Test queries
    test_queries = [
        "What is our financial performance in Q1 2024?",
        "What are the main customer complaints?",
        "What are our competitors doing?"
    ]
    
    for query in test_queries:
        result = rag.query(query, top_k=3)
        rag.display_result(result, show_sources=True)
        print("\n")