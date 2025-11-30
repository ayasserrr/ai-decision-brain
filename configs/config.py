import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for RAG system"""
    
    # Project paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "Data"
    DOCUMENTS_DIR = DATA_DIR / "documents"
    VECTOR_DB_DIR = DATA_DIR / "vector_db"
    
    # Embedding model configuration
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384 
    
    # LLM configuration - Load from environment
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY").strip()
    OPENAI_MODEL = os.getenv("OPENAI_MODEL")
    OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
    
    # Chunking configuration
    CHUNK_SIZE = 500  
    CHUNK_OVERLAP = 50  

    # Retrieval configuration
    TOP_K_RESULTS = 5 
    SIMILARITY_THRESHOLD = 0.5  
    
    # Generation configuration
    MAX_TOKENS = 1000  
    TEMPERATURE = 0.7  
    
    @classmethod
    def has_openai_key(cls):
        """Check if OpenAI API key is available and valid"""
        return bool(cls.OPENAI_API_KEY and len(cls.OPENAI_API_KEY) > 20)
    
    @classmethod
    def get_llm_config(cls):
        """Get LLM configuration based on available credentials"""
        if cls.has_openai_key():
            # OpenAI is available
            return {
                'provider': 'openai',
                'api_url': 'https://api.openai.com/v1/',
                'api_key': cls.OPENAI_API_KEY,
                'model': cls.OPENAI_MODEL,
                'fallback': {
                    'provider': 'ollama',
                    'api_url': cls.OLLAMA_API_URL,
                    'api_key': 'ollama',
                    'model': cls.OLLAMA_MODEL
                }
            }
        else:
            # Use Ollama only
            return {
                'provider': 'ollama',
                'api_url': cls.OLLAMA_API_URL,
                'api_key': 'ollama',
                'model': cls.OLLAMA_MODEL
            }
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.DOCUMENTS_DIR.mkdir(exist_ok=True)
        cls.VECTOR_DB_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def validate_config(cls):
        """Validate that all required configurations are set"""
        assert cls.DOCUMENTS_DIR.exists(), f"Documents directory not found: {cls.DOCUMENTS_DIR}"
        
        llm_config = cls.get_llm_config()
        print("Configuration validated successfully")
        print(f"LLM Provider: {llm_config['provider'].upper()}")
        print(f"Model: {llm_config['model']}")
        
        if 'fallback' in llm_config:
            print(f"Fallback: {llm_config['fallback']['provider'].upper()} available")
    
    @classmethod
    def display_config(cls):
        """Display current configuration (hiding sensitive info)"""
        print("=" * 50)
        print("RAG System Configuration")
        print("=" * 50)
        print(f"Documents Directory: {cls.DOCUMENTS_DIR}")
        print(f"Vector DB Directory: {cls.VECTOR_DB_DIR}")
        print(f"Embedding Model: {cls.EMBEDDING_MODEL_NAME}")
        
        llm_config = cls.get_llm_config()
        print(f"\nLLM Configuration:")
        print(f"  Primary Provider: {llm_config['provider'].upper()}")
        print(f"  Primary Model: {llm_config['model']}")
        
        if llm_config['provider'] == 'openai':
            print(f"  OpenAI Key: {'*' * 8}...{cls.OPENAI_API_KEY[-4:]}")
        else:
            print(f"  Ollama URL: {cls.OLLAMA_API_URL}")
        
        if 'fallback' in llm_config:
            print(f"  Fallback Provider: {llm_config['fallback']['provider'].upper()}")
            print(f"  Fallback Model: {llm_config['fallback']['model']}")
        
        print(f"\nRetrieval Settings:")
        print(f"  Chunk Size: {cls.CHUNK_SIZE}")
        print(f"  Chunk Overlap: {cls.CHUNK_OVERLAP}")
        print(f"  Top K Results: {cls.TOP_K_RESULTS}")
        print("=" * 50)


# Create directories on import
Config.ensure_directories()