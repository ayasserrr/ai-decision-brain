# AI Decision Brain

> **Transform Data into Strategic Decisions**  
> *AI-powered intelligence platform that helps executives make data-driven decisions based on company data and internal knowledge*

---

## What is AI Decision Brain?

AI Decision Brain is an advanced Retrieval-Augmented Generation (RAG) system designed specifically for **executive decision support**. It transforms your company's internal documents, reports, and data into **actionable strategic recommendations** with clear financial impact and implementation roadmaps.

### Why It's Powerful?

**Core Value Proposition**: Convert complex data into strategic decisions ready for immediate execution.

---

## Key Features

### Strategic Decision Intelligence
- **Executive Decision Support**: Get AI-powered strategic recommendations
- **Financial Impact Analysis**: Quantitative estimates with dollar-specific impacts
- **Risk Assessment**: Identify and mitigate potential risks
- **Implementation Roadmaps**: Phased execution plans with timelines

### Advanced RAG Capabilities
- **PDF Document Processing**: Advanced PDF text extraction and analysis
- **Smart Vector Search**: Semantic understanding of business context
- **Relevance Scoring**: Intelligent document prioritization
- **Source Citation**: Trace recommendations back to original data

### Flexible AI Integration
- **OpenAI GPT Support**: Primary high-performance analysis
- **Ollama Local Models**: Free, private fallback option
- **Automatic Failover**: Seamless switching between AI providers
- **Customizable Models**: Configurable for different use cases

---

## Project Structure

```
ai-decision-brain/
├── Data/
│   ├── documents/          # Your business documents
│   └── vector_db/          # Generated vector database
├── src/
│   ├── rag_pipeline.py     # Main RAG pipeline
│   ├── embedding_generator.py
│   ├── vector_store.py
│   ├── document_loader.py
│   ├── text_chunker.py
│   └── __init__.py
├── configs/
│   └── config.py          # System configuration
├── api.py                  # FastAPI web server
├── app.py                  # Streamlit web interface
├── main.py                 # CLI interface
├── .env                    # Environment variables
├── .env.example            # Configuration template
└── requirements.txt        # Python dependencies
```

---

## Installation

### Prerequisites

- Python 3.8+
- 2GB+ RAM
- 500MB+ storage space

### 1. Clone and Install Dependencies

```bash
# Clone the repository
git clone https://github.com/ayasserrr/ai-decision-brain.git
cd ai-decision-brain

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Copy the environment template and configure your settings:

```bash
cp .env.example .env
```

Edit `.env` file with your preferred AI provider:

```env
# =============================================================================
# LLM Configuration
# =============================================================================

# OpenAI Configuration (Primary - if you have API key)
# Get your API key from: https://platform.openai.com/api-keys
# Leave empty if you don't have one
OPENAI_API_KEY=your_openai_api_key_here

# OpenAI Model (only used if OPENAI_API_KEY is set)
OPENAI_MODEL=gpt-3.5-turbo

# Ollama Configuration (Fallback - Free Local LLM)
OLLAMA_API_URL=http://localhost:11434/v1/
OLLAMA_MODEL=llama2
```

### 3. Optional: Install Ollama for Local AI

If you want to use free local AI models:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama2

# Start Ollama server
ollama serve
```

### 4. Add Your Documents

Place your business documents in the `Data/documents/` folder:
- Financial reports (PDF)
- Meeting notes (PDF)
- Market research (PDF)
- Performance data (PDF)
- Strategic plans (PDF)

**Note**: Currently only PDF documents are supported. 

---

## Quick Start

### Method 1: Web Interface (Recommended)

```bash
# Start the API server
python api.py

# In another terminal, start the web interface
streamlit run app.py
```

Open `http://localhost:8501` in your browser to access the web interface.

### Method 2: Direct Python Usage

```python
from src.rag_pipeline import StrategicRAGPipeline

# Initialize the system
rag = StrategicRAGPipeline()

# Ask strategic questions
result = rag.query("What is the best decision to reduce customer churn rate?")
rag.display_strategic_result(result)
```

### Method 3: Command Line

```bash
# Initialize the system and test with sample queries
python src/rag_pipeline.py
```

---

## How It Works

### System Flow

1. **Document Ingestion** → Process internal company documents
2. **Knowledge Embedding** → Create semantic vector representations using sentence-transformers
3. **Intelligent Retrieval** → Find relevant information for strategic questions
4. **Strategic Analysis** → Generate data-driven recommendations
5. **Decision Output** → Deliver actionable insights with evidence

### Example Strategic Questions

Ask questions like:
- *"What is the impact of a 10% cost increase on next quarter's performance?"*
- *"What is the best decision to reduce customer churn rate?"*
- *"How can we improve operational efficiency based on Q3 data?"*
- *"What strategic moves should we make to counter competitor activities?"*

### Sample Output Structure

```
STRATEGIC DECISION SUPPORT REPORT
===================================

QUESTION: What is the best decision to reduce customer churn rate?

EXECUTIVE SUMMARY
===================================
**Decision Recommendation**: Implement a proactive customer retention program
**Key Impact**: Expected 15% reduction in churn, 8% revenue increase
**Urgency Level**: High - Current churn rate exceeds industry benchmarks
**Bottom Line**: Immediate implementation recommended with strong ROI

DETAILED STRATEGIC ANALYSIS
===================================
[Comprehensive analysis with supporting data and recommendations]

SUPPORTING EVIDENCE SOURCES
===================================
1. Customer_Feedback_Summary.pdf (Similarity: 92.3%)
2. Employee_Satisfaction_Survey.pdf (Similarity: 87.1%)
3. Competitor_Analysis_2024.pdf (Similarity: 78.4%)
```

---

## Configuration Options

### AI Providers

#### OpenAI GPT (Recommended for Production)
- Higher accuracy and better reasoning
- Requires API key
- Costs apply based on usage

#### Ollama Local (Free Alternative)
- Complete privacy and no API costs
- Requires local installation
- Good for exploration and testing

### Performance Settings

Default configuration in `configs/config.py`:

```python
# Embedding settings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Document processing
CHUNK_SIZE = 500          # Characters per chunk
CHUNK_OVERLAP = 50        # Overlap between chunks

# Retrieval settings
TOP_K_RESULTS = 5         # Most relevant documents to retrieve
SIMILARITY_THRESHOLD = 0.5  # Minimum relevance score

# Generation settings
MAX_TOKENS = 1000         # Maximum response length
TEMPERATURE = 0.7         # Creativity vs consistency balance
```

---

## API Reference

### Web API Endpoints

Start the API server with `python api.py` then access:

#### Health Check
```http
GET /health
```

Returns system status and statistics.

#### Query System
```http
POST /query
Content-Type: application/json

{
  "question": "What is our Q1 2024 financial performance?",
  "top_k": 5,
  "return_context": false
}
```

#### System Stats
```http
GET /stats
```

Returns configuration and vector database statistics.

### API Documentation

When the API server is running, visit `http://localhost:8000/docs` for interactive API documentation.

---

## Supported Document Types

- **PDF Documents**: All PDF-based reports and documents
  - Financial reports
  - Meeting notes
  - Market research
  - Performance data
  - Strategic plans
  - Customer feedback

**Current Limitation**: Only PDF format is currently supported.

---

## Advanced Usage

### Custom Configuration

```python
from configs import Config

# Display current configuration
Config.display_config()

# Validate setup
Config.validate_config()
```

### Batch Processing

```python
# Process multiple questions
questions = [
    "What is our financial outlook for Q2?",
    "How can we improve operational efficiency?",
    "What are the main competitive threats?"
]

for question in questions:
    result = rag.query(question, top_k=8, return_context=True)
    print(f"Question: {question}")
    print(f"Summary: {result['executive_summary']}")
    print("-" * 50)
```

### Accessing Raw Results

```python
result = rag.query("Strategic question here")

# Access all components
print("Executive Summary:")
print(result['executive_summary'])

print("\nDetailed Analysis:")
print(result['detailed_analysis'])

print("\nSources:")
for source in result['sources']:
    print(f"- {source['filename']} (Similarity: {source['similarity']})")

print(f"\nProvider: {result['provider']}")
print(f"Model: {result['model']}")
print(f"Tokens Used: {result['tokens_used']}")
```

---

## Troubleshooting

### Common Issues

#### Ollama Connection Failed
```bash
# Install and start Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama2
ollama serve
```

#### Document Processing Errors
- Ensure documents are not password protected
- Check file formats are supported
- Verify document text is extractable

#### Vector Database Issues
```bash
# Rebuild vector database
rm -rf Data/vector_db/*
python src/rag_pipeline.py
```

#### Import Errors
```bash
# Ensure you're in the project root
cd ai-decision-brain
python -c "from src.rag_pipeline import StrategicRAGPipeline; print('Import successful')"
```

### Performance Tips

1. **Document Quality**: Use well-structured documents for better analysis
2. **Specific Questions**: Ask focused, strategic questions
3. **Regular Updates**: Refresh documents quarterly for current insights
4. **Model Selection**: Use OpenAI for critical decisions, Ollama for exploration

---

## Technical Architecture

### Core Components

- **StrategicRAGPipeline**: Main orchestration class
- **EmbeddingGenerator**: Creates vector embeddings using sentence-transformers
- **VectorStore**: FAISS-based vector similarity search
- **DocumentLoader**: PDF document processing using PyPDF2
- **TextChunker**: Intelligent text segmentation

### Dependencies

Key libraries from `requirements.txt`:
- **fastapi**: Web API framework
- **streamlit**: Web interface
- **openai**: AI model integration
- **sentence-transformers**: Text embeddings
- **faiss-cpu**: Vector similarity search
- **PyPDF2**: PDF document processing

---

## Use Cases

### Executive Teams
- Strategic planning sessions
- Investment decision support
- Operational efficiency analysis
- Competitive response planning

### Finance Departments
- Budget optimization
- Cost-benefit analysis
- Financial forecasting support
- Risk assessment

### Marketing Teams
- Campaign performance analysis
- Budget allocation optimization
- Customer retention strategies
- Market opportunity identification

---

## Privacy & Security

- **Local Processing**: All document processing happens locally
- **Optional Cloud AI**: OpenAI API usage is optional
- **Data Ownership**: You retain all rights to your data
- **No Data Sharing**: Documents never leave your system without explicit API usage

---

## Contributing

We welcome contributions! Feel free to submit pull requests or open issues for improvements and bug fixes.

### Development Setup
```bash
git clone https://github.com/ayasserrr/ai-decision-brain.git
cd ai-decision-brain
pip install -r requirements.txt
```

## Author

**Aya Yasser** - Email: ayasser.tawfik@gmail.com - Phone: +20 1025027056

---

## License

This project is licensed under the MIT License.

---

## Get Started Today

Transform your company data into strategic decisions:

1. **Add your documents** to `Data/documents/` 
2. **Configure your AI preferences** in `.env` 
3. **Start the system**: `python api.py` and `streamlit run app.py`
4. **Start making better decisions** with AI-powered insights!

---

**AI Decision Brain** - *Where Data Meets Decision Intelligence*