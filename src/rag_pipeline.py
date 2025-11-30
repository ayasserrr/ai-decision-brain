import sys
from pathlib import Path
import openai
from typing import List, Dict, Tuple
from datetime import datetime

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


class StrategicRAGPipeline:
    """Strategic RAG pipeline for executive decision support based on company data and internal knowledge"""
    
    def __init__(self, vector_store_path: str = None):
        # Get LLM configuration
        self.llm_config = Config.get_llm_config()
        
        # Initialize primary client
        print("Initializing Strategic RAG Pipeline...")
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
        print("Strategic RAG Pipeline initialized successfully")
    
    def retrieve_context(self, query: str, top_k: int = None) -> List[Tuple[int, float, Dict]]:
        """Retrieve relevant context for a strategic query"""
        top_k = top_k or Config.TOP_K_RESULTS
        
        # Generate query embedding
        query_embedding = self.embedder.generate_embedding(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=top_k, normalize=True)
        
        return results
    
    def format_context(self, results: List[Tuple[int, float, Dict]]) -> str:
        """Format retrieved chunks into detailed context string"""
        context_parts = []
        
        for rank, (idx, similarity, metadata) in enumerate(results, 1):
            context_parts.append(
                f"=== DOCUMENT {rank}: {metadata['filename']} (Relevance: {similarity:.2%}) ===\n"
                f"Content: {metadata['text']}\n"
            )
        
        return "\n".join(context_parts)
    
    def create_strategic_prompt(self, query: str, context: str) -> str:
        """Create strategic decision-making prompt for the LLM"""
        prompt = f"""You are a Chief Strategy Officer AI assistant. Your role is to transform data into actionable strategic decisions.

COMPANY DATA & CONTEXT:
{context}

STRATEGIC QUESTION: {query}

ANALYZE THIS SITUATION AND PROVIDE:

SUGGESTED DECISION:
[Clear, actionable recommendation that directly addresses the strategic question]

EXPECTED IMPACT:
- Financial Impact: [Quantitative estimate if possible]
- Operational Impact: [How this affects operations]
- Timeline: [Expected implementation and results timeline]
- Key Metrics Affected: [Specific KPIs that will change]

SUPPORTING EVIDENCE:
- Source 1: [Document name] - [How this evidence supports the decision]
- Source 2: [Document name] - [Key insight from this source]
- Source 3: [Document name] - [Relevant data point or finding]

RISKS & MITIGATIONS:
- Primary Risks: [What could go wrong]
- Mitigation Strategies: [How to address these risks]
- Success Factors: [Critical elements for success]

IMPLEMENTATION ROADMAP:
- Phase 1 (Immediate): [First 30 days actions]
- Phase 2 (Short-term): [Next 60 days]
- Phase 3 (Medium-term): [Following 90 days]

CRITICAL REQUIREMENTS:
- Base your analysis ONLY on the provided company data
- Be specific and actionable - avoid vague recommendations
- Consider financial, operational, and strategic dimensions
- Prioritize decisions with measurable impact
- Address both opportunities and risks

Remember: You are helping executives make data-driven strategic decisions."""

        return prompt
    
    def create_executive_summary(self, detailed_analysis: str) -> str:
        """Create concise executive summary from detailed analysis"""
        summary_prompt = f"""Create a concise executive summary from this detailed strategic analysis:

{detailed_analysis}

EXECUTIVE SUMMARY FORMAT:

**Decision Recommendation**: [One sentence summary]

**Key Impact**: [Top 2-3 expected outcomes]

**Urgency Level**: [High/Medium/Low - with justification]

**Bottom Line**: [Overall recommendation and expected value]"""

        return summary_prompt
    
    def _call_llm(self, client, model: str, prompt: str, is_summary: bool = False) -> Dict[str, any]:
        """Internal method to call LLM API for strategic analysis"""
        system_message = (
            "You are a concise executive assistant creating summaries." 
            if is_summary else 
            "You are a Chief Strategy Officer AI with expertise in data-driven decision making, financial analysis, and strategic planning. You transform complex data into clear, actionable executive decisions."
        )
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            max_tokens=Config.MAX_TOKENS,
            temperature=0.3 if is_summary else 0.7  # Lower temperature for summaries
        )
        
        answer = response.choices[0].message.content
        tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else None
        
        return {
            'answer': answer,
            'tokens_used': tokens_used
        }
    
    def generate_strategic_analysis(self, query: str, context: str) -> Dict[str, any]:
        """Generate strategic analysis using the LLM with automatic fallback"""
        prompt = self.create_strategic_prompt(query, context)
        
        # Try primary LLM
        try:
            result = self._call_llm(
                client=self.primary_client,
                model=self.llm_config['model'],
                prompt=prompt
            )
            
            # Generate executive summary
            summary_prompt = self.create_executive_summary(result['answer'])
            summary_result = self._call_llm(
                client=self.primary_client,
                model=self.llm_config['model'],
                prompt=summary_prompt,
                is_summary=True
            )
            
            return {
                'detailed_analysis': result['answer'],
                'executive_summary': summary_result['answer'],
                'model': self.llm_config['model'],
                'provider': self.llm_config['provider'],
                'tokens_used': result.get('tokens_used', 0) + summary_result.get('tokens_used', 0),
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
                    
                    # Generate executive summary with fallback
                    summary_prompt = self.create_executive_summary(result['answer'])
                    summary_result = self._call_llm(
                        client=self.fallback_client,
                        model=self.llm_config['fallback']['model'],
                        prompt=summary_prompt,
                        is_summary=True
                    )
                    
                    print(f"[SUCCESS] Fallback strategic analysis completed!\n")
                    
                    return {
                        'detailed_analysis': result['answer'],
                        'executive_summary': summary_result['answer'],
                        'model': self.llm_config['fallback']['model'],
                        'provider': self.llm_config['fallback']['provider'],
                        'tokens_used': result.get('tokens_used', 0) + summary_result.get('tokens_used', 0),
                        'fallback_used': True
                    }
                
                except Exception as fallback_error:
                    return {
                        'detailed_analysis': f"Error: Both LLM providers failed.\n\nPrimary ({self.llm_config['provider'].upper()}): {error_msg}\n\nFallback ({self.llm_config['fallback']['provider'].upper()}): {str(fallback_error)}",
                        'executive_summary': "Unable to generate executive summary due to system errors.",
                        'model': None,
                        'provider': None,
                        'tokens_used': None,
                        'error': True
                    }
            else:
                return {
                    'detailed_analysis': f"Error generating strategic analysis: {error_msg}",
                    'executive_summary': "Unable to generate executive summary.",
                    'model': self.llm_config['model'],
                    'provider': self.llm_config['provider'],
                    'tokens_used': None,
                    'error': True
                }
    
    def query(self, question: str, top_k: int = None, return_context: bool = False) -> Dict[str, any]:
        """Complete strategic RAG query pipeline"""
        print(f"\nSTRATEGIC QUERY: {question}")
        print("=" * 60)
        
        # Retrieve context
        print("Retrieving relevant business context...")
        results = self.retrieve_context(question, top_k=top_k)
        
        print(f"Found {len(results)} relevant business documents")
        
        # Format context
        context = self.format_context(results)
        
        # Generate strategic analysis
        print("Generating strategic analysis...")
        response = self.generate_strategic_analysis(question, context)
        
        # Prepare comprehensive output
        output = {
            'question': question,
            'executive_summary': response['executive_summary'],
            'detailed_analysis': response['detailed_analysis'],
            'sources': [
                {
                    'filename': metadata['filename'],
                    'chunk_id': metadata['chunk_id'],
                    'similarity': similarity,
                    'relevance': f"{similarity:.2%}"
                }
                for idx, similarity, metadata in results
            ],
            'model': response.get('model'),
            'provider': response.get('provider'),
            'tokens_used': response.get('tokens_used'),
            'fallback_used': response.get('fallback_used', False),
            'error': response.get('error', False),
            'timestamp': datetime.now().isoformat()
        }
        
        if return_context:
            output['context'] = context
            output['retrieved_chunks'] = [
                {
                    'text': metadata['text'],
                    'filename': metadata['filename'],
                    'similarity': similarity,
                    'relevance': f"{similarity:.2%}"
                }
                for idx, similarity, metadata in results
            ]
        
        return output
    
    def display_strategic_result(self, result: Dict[str, any], show_sources: bool = True):
        """Display strategic query result in an executive-friendly format"""
        print("\n" + "=" * 70)
        print("STRATEGIC DECISION SUPPORT REPORT")
        print("=" * 70)
        
        print(f"\nQUESTION: {result['question']}")
        print(f"Analysis Date: {result['timestamp']}")
        
        print("\n" + "=" * 70)
        print("EXECUTIVE SUMMARY")
        print("=" * 70)
        print(result['executive_summary'])
        
        print("\n" + "=" * 70)
        print("DETAILED STRATEGIC ANALYSIS")
        print("=" * 70)
        print(result['detailed_analysis'])
        
        if show_sources and not result.get('error'):
            print("\n" + "=" * 70)
            print("SUPPORTING EVIDENCE SOURCES")
            print("=" * 70)
            for i, source in enumerate(result['sources'], 1):
                print(f"{i}. {source['filename']} (Relevance: {source['relevance']})")
        
        print(f"\nAnalysis Provider: {result.get('provider', 'N/A').upper() if result.get('provider') else 'N/A'}")
        print(f"Model: {result.get('model', 'N/A') if result.get('model') else 'N/A'}")
        
        if result.get('fallback_used'):
            print("Note: Fallback LLM was used for this analysis")
        
        if result.get('tokens_used'):
            print(f"Analysis Depth: {result['tokens_used']} tokens")
        
        print("=" * 70)


# Test function with strategic queries
if __name__ == "__main__":
    # Initialize Strategic RAG pipeline
    print("Initializing AI Decision Brain Strategic Decision Support System...")
    rag = StrategicRAGPipeline()
    
    # Strategic test queries
    strategic_queries = [
        "What is the impact of a 10% cost increase on next quarter's performance?",
        "What is the best decision to reduce customer churn rate?",
        "How can we improve operational efficiency based on Q3 data?",
        "What strategic moves should we make to counter competitor activities?",
        "How can we optimize our marketing budget for maximum ROI?"
    ]
    
    print(f"\nTesting Strategic Decision Support with {len(strategic_queries)} queries...")
    
    for i, query in enumerate(strategic_queries, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(strategic_queries)}")
        print(f"{'='*80}")
        
        result = rag.query(query, top_k=5)
        rag.display_strategic_result(result, show_sources=True)
        
        # Add spacing between tests
        if i < len(strategic_queries):
            print("\n" + "->" * 20 + " NEXT ANALYSIS " + "<-" * 20 + "\n")