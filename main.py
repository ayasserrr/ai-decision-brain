import sys
from pathlib import Path
from configs import Config
from src import StrategicRAGPipeline


def print_header():
    """Print application header"""
    print("\n" + "=" * 60)
    print("AI DECISION BRAIN - RAG System")
    print("=" * 60)
    print("Ask questions about your documents")
    print("Commands: 'quit' or 'exit' to stop, 'help' for help")
    print("=" * 60 + "\n")


def print_help():
    """Print help information"""
    print("\n" + "-" * 60)
    print("HELP - Available Commands")
    print("-" * 60)
    print("- Type your question and press Enter to get an answer")
    print("- 'quit' or 'exit' - Exit the application")
    print("- 'help' - Show this help message")
    print("- 'stats' - Show system statistics")
    print("- 'clear' - Clear the screen")
    print("-" * 60 + "\n")


def print_stats(rag: StrategicRAGPipeline):
    """Print system statistics"""
    stats = rag.vector_store.get_stats()
    
    print("\n" + "-" * 60)
    print("SYSTEM STATISTICS")
    print("-" * 60)
    print(f"Status: {stats['status']}")
    print(f"Total Documents Indexed: {stats['total_vectors']} chunks")
    print(f"Embedding Dimension: {stats['dimension']}")
    llm_config = Config.get_llm_config()
    print(f"LLM Model: {llm_config['model']}")
    print(f"Embedding Model: {Config.EMBEDDING_MODEL_NAME}")
    print("-" * 60 + "\n")


def clear_screen():
    """Clear the terminal screen"""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')


def interactive_mode():
    """Run interactive question-answering mode"""
    try:
        # Initialize RAG pipeline
        print("\nInitializing system...")
        rag = StrategicRAGPipeline()
        
        # Print header
        print_header()
        
        # Main loop
        while True:
            try:
                # Get user input
                user_input = input("\nYour question: ").strip()
                
                # Handle empty input
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit']:
                    print("\nThank you for using AI Decision Brain. Goodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    print_help()
                    continue
                
                elif user_input.lower() == 'stats':
                    print_stats(rag)
                    continue
                
                elif user_input.lower() == 'clear':
                    clear_screen()
                    print_header()
                    continue
                
                # Process question
                print("\nProcessing your question...")
                result = rag.query(user_input, top_k=Config.TOP_K_RESULTS)
                
                # Display result
                print("\n" + "=" * 60)
                print("EXECUTIVE SUMMARY")
                print("=" * 60)
                print(result['executive_summary'])
                
                print("\n" + "=" * 60)
                print("DETAILED ANALYSIS")
                print("=" * 60)
                print(result['detailed_analysis'])
                
                print("\n" + "-" * 60)
                print("SOURCES")
                print("-" * 60)
                for i, source in enumerate(result['sources'], 1):
                    print(f"{i}. {source['filename']} (Relevance: {source['similarity']:.2%})")
                
                if result.get('tokens_used'):
                    print(f"\nTokens used: {result['tokens_used']}")
                
                print("-" * 60)
            
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'quit' to exit or continue asking questions.")
                continue
            
            except Exception as e:
                print(f"\nError processing question: {str(e)}")
                print("Please try again or type 'help' for assistance.")
    
    except Exception as e:
        print(f"\nFailed to initialize system: {str(e)}")
        print("Please ensure:")
        print("1. Vector store exists (run setup first)")
        print("2. LLM server is running")
        print("3. All dependencies are installed")
        sys.exit(1)


def single_question_mode(question: str):
    """Answer a single question and exit"""
    try:
        print("\nInitializing system...")
        rag = StrategicRAGPipeline()
        
        print(f"\nQuestion: {question}")
        print("-" * 60)
        
        result = rag.query(question, top_k=Config.TOP_K_RESULTS)
        
        print("\nAnswer:")
        print(result['executive_summary'])
        print("\n\nDetailed Analysis:")
        print(result['detailed_analysis'])
        
        print("\n\nSources:")
        for i, source in enumerate(result['sources'], 1):
            print(f"{i}. {source['filename']} (Relevance: {source['similarity']:.2%})")
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)


def main():
    """Main entry point"""
    # Check if vector store exists
    if not (Config.VECTOR_DB_DIR / "faiss_index.bin").exists():
        print("\nERROR: Vector store not found!")
        print("Please run the setup first:")
        print("  python src/rag_pipeline.py")
        sys.exit(1)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        # Single question mode
        question = " ".join(sys.argv[1:])
        single_question_mode(question)
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()