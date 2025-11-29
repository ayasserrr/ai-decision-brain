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
    
    # LLM configuration 
    OPENAI_API_URL = os.getenv("OPENAI_API_URL")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GENERATION_MODEL_ID = os.getenv("GENERATION_MODEL_ID")
    
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
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.DOCUMENTS_DIR.mkdir(exist_ok=True)
        cls.VECTOR_DB_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def validate_config(cls):
        """Validate that all required configurations are set"""
        assert cls.DOCUMENTS_DIR.exists(), f"Documents directory not found: {cls.DOCUMENTS_DIR}"
        assert cls.OPENAI_API_URL, "OPENAI_API_URL not set in .env file"
        assert cls.OPENAI_API_KEY, "OPENAI_API_KEY not set in .env file"
        assert cls.GENERATION_MODEL_ID, "GENERATION_MODEL_ID not set in .env file"
        print("✓ Configuration validated successfully")
    
    @classmethod
    def display_config(cls):
        """Display current configuration (hiding sensitive info)"""
        print("=" * 50)
        print("RAG System Configuration")
        print("=" * 50)
        print(f"Documents Directory: {cls.DOCUMENTS_DIR}")
        print(f"Vector DB Directory: {cls.VECTOR_DB_DIR}")
        print(f"Embedding Model: {cls.EMBEDDING_MODEL_NAME}")
        print(f"LLM API URL: {cls.OPENAI_API_URL if cls.OPENAI_API_URL else '[NOT SET]'}")
        print(f"LLM Model: {cls.GENERATION_MODEL_ID if cls.GENERATION_MODEL_ID else '[NOT SET]'}")
        print(f"API Key: {'*' * 8 if cls.OPENAI_API_KEY else '[NOT SET]'}")
        print(f"Chunk Size: {cls.CHUNK_SIZE}")
        print(f"Top K Results: {cls.TOP_K_RESULTS}")
        print("=" * 50)


# Create directories on import
Config.ensure_directories()