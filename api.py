from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import uvicorn
from datetime import datetime
import os

from configs import Config
from src import StrategicRAGPipeline
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_pipeline
    rag_pipeline = StrategicRAGPipeline()
    yield

# Initialize FastAPI app
app = FastAPI(
    title="AI Decision Brain API",
    description="RAG-based Question Answering API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG pipeline instance
rag_pipeline = None


# Pydantic models
class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask", min_length=1)
    top_k: Optional[int] = Field(None, description="Number of chunks to retrieve", ge=1, le=20)
    return_context: Optional[bool] = Field(False, description="Whether to return retrieved context")


class Source(BaseModel):
    filename: str
    chunk_id: int
    similarity: float


class QueryResponse(BaseModel):
    question: str
    executive_summary: str
    detailed_analysis: str
    sources: List[Source]
    timestamp: str
    tokens_used: Optional[int] = None
    context: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    fallback_used: Optional[bool] = None
    error: Optional[bool] = None


class HealthResponse(BaseModel):
    status: str
    message: str
    vector_store_status: str
    total_chunks: int
    embedding_dimension: int
    llm_model: str


class ErrorResponse(BaseModel):
    error: str
    detail: str


# Routes
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to AI Decision Brain API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "query": "/query (POST)",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        stats = rag_pipeline.vector_store.get_stats()
        
        return HealthResponse(
            status="healthy",
            message="System is operational",
            vector_store_status=stats['status'],
            total_chunks=stats.get('total_vectors', 0),
            embedding_dimension=stats.get('dimension', 0),
            llm_model=rag_pipeline.llm_config['model']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest):
    """Query the RAG system"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        # Process query
        result = rag_pipeline.query(
            question=request.question,
            top_k=request.top_k or Config.TOP_K_RESULTS,
            return_context=request.return_context
        )
        
        # Format response
        response = QueryResponse(
            question=result['question'],
            executive_summary=result['executive_summary'],
            detailed_analysis=result['detailed_analysis'],
            sources=[
                Source(
                    filename=source['filename'],
                    chunk_id=source['chunk_id'],
                    similarity=source['similarity']
                )
                for source in result['sources']
            ],
            timestamp=result['timestamp'],
            tokens_used=result.get('tokens_used'),
            context=result.get('context') if request.return_context else None,
            provider=result.get('provider'),
            model=result.get('model'),
            fallback_used=result.get('fallback_used'),
            error=result.get('error')
        )
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/stats", tags=["Info"])
async def get_stats():
    """Get system statistics"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        stats = rag_pipeline.vector_store.get_stats()
        
        return {
            "vector_store": stats,
            "config": {
                "embedding_model": Config.EMBEDDING_MODEL_NAME,
                "llm_model": rag_pipeline.llm_config['model'],
                "chunk_size": Config.CHUNK_SIZE,
                "chunk_overlap": Config.CHUNK_OVERLAP,
                "top_k_results": Config.TOP_K_RESULTS
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "status_code": 500
        }
    )


# Run server
if __name__ == "__main__":
    print("Starting AI Decision Brain API...")
    print(f"API Documentation: http://localhost:8000/docs")
    print(f"Health Check: http://localhost:8000/health")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=(os.getenv("UVICORN_RELOAD", "0") == "1")
    )