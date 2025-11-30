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
    """Complete RAG pipeline for question answering with automatic OpenAI/Ollama fallback"""
    
    def __init__(self, vector_store_path: str = None):
        # Get LLM configuration
        self.llm_config = Config.get_llm_config()
        
        # Initialize primary client
        print("Initializing RAG Pipeline...")
        print("-" * 50)
        print(f"Primary LLM: {self.llm_config['provider'].upper()}")
        print(f"Primary Model: {self.llm_config['model']}")
        
        self.primary_client = openai.OpenAI(
            base_url=self.llm_config['api_url'],
            api_key=self.llm_config['api_key']
        )
        
        # Initialize fallback client if available
        self.fallback_client = None
        if 'fallback' in self.llm_config:
            print(f"Fallback LLM: {self.llm_config['fallback']['provider'].upper()}")
            print(f"Fallback Model: {self.llm_config['fallback']['model']}")
            self.fallback_client = openai.OpenAI(
                base_url=self.llm_config['fallback']['api_url'],
                api_key=self.llm_config['fallback']['api_key']
            )
        
        # Initialize components
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
    
    def _call_llm(self, client, model: str, prompt: str) -> Dict[str, any]:
        """Internal method to call LLM API"""
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=Config.MAX_TOKENS,
            temperature=Config.TEMPERATURE
        )
        
        answer = response.choices[0].message.content
        tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else None
        
        return {
            'answer': answer,
            'tokens_used': tokens_used
        }
    
    def generate_answer(self, query: str, context: str) -> Dict[str, any]:
        """Generate answer using the LLM with automatic fallback"""
        prompt = self.create_prompt(query, context)
        
        # Try primary LLM
        try:
            result = self._call_llm(
                client=self.primary_client,
                model=self.llm_config['model'],
                prompt=prompt
            )
            
            return {
                'answer': result['answer'],
                'model': self.llm_config['model'],
                'provider': self.llm_config['provider'],
                'tokens_used': result.get('tokens_used'),
                'fallback_used': False
            }
        
        except Exception as primary_error:
            error_msg = str(primary_error)
            print(f"\n[WARNING] Primary LLM ({self.llm_config['provider'].upper()}) failed: {error_msg}")
            
            # Try fallback if available
            if self.fallback_client:
                print(f"[INFO] Attempting fallback to {self.llm_config['fallback']['provider'].upper()}...")
                
                try:
                    result = self._call_llm(
                        client=self.fallback_client,
                        model=self.llm_config['fallback']['model'],
                        prompt=prompt
                    )
                    
                    print(f"[SUCCESS] Fallback succeeded!\n")
                    
                    return {
                        'answer': result['answer'],
                        'model': self.llm_config['fallback']['model'],
                        'provider': self.llm_config['fallback']['provider'],
                        'tokens_used': result.get('tokens_used'),
                        'fallback_used': True
                    }
                
                except Exception as fallback_error:
                    return {
                        'answer': f"Error: Both LLM providers failed.\n\nPrimary ({self.llm_config['provider'].upper()}): {error_msg}\n\nFallback ({self.llm_config['fallback']['provider'].upper()}): {str(fallback_error)}",
                        'model': None,
                        'provider': None,
                        'tokens_used': None,
                        'error': True
                    }
            else:
                return {
                    'answer': f"Error generating answer: {error_msg}",
                    'model': self.llm_config['model'],
                    'provider': self.llm_config['provider'],
                    'tokens_used': None,
                    'error': True
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
            'model': response.get('model'),
            'provider': response.get('provider'),
            'tokens_used': response.get('tokens_used'),
            'fallback_used': response.get('fallback_used', False),
            'error': response.get('error', False)
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
        
        if show_sources and not result.get('error'):
            print("\n" + "=" * 50)
            print("SOURCES")
            print("=" * 50)
            for i, source in enumerate(result['sources'], 1):
                print(f"{i}. {source['filename']} (Similarity: {source['similarity']:.4f})")
        
        print(f"\nProvider: {result.get('provider', 'N/A').upper() if result.get('provider') else 'N/A'}")
        print(f"Model: {result.get('model', 'N/A') if result.get('model') else 'N/A'}")
        
        if result.get('fallback_used'):
            print("[INFO] Fallback LLM was used")
        
        if result.get('tokens_used'):
            print(f"Tokens used: {result['tokens_used']}")
        
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