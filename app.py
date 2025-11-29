import streamlit as st
import requests
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="AI Decision Brain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .question-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .answer-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #0068c9;
    }
    .source-item {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
        border-left: 3px solid #28a745;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if API is healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except Exception as e:
        return False, str(e)


def query_rag(question, top_k=5, return_context=False):
    """Query the RAG API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={
                "question": question,
                "top_k": top_k,
                "return_context": return_context
            },
            timeout=60
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json()
    except Exception as e:
        return False, str(e)


def get_stats():
    """Get system statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 AI Decision Brain</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem;">Ask questions about your documents</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Health Check
        st.subheader("🔌 API Status")
        is_healthy, health_data = check_api_health()
        
        if is_healthy:
            st.success("✓ API Connected")
            if health_data:
                st.metric("Total Chunks", health_data.get('total_chunks', 'N/A'))
                st.metric("Embedding Dimension", health_data.get('embedding_dimension', 'N/A'))
        else:
            st.error("✗ API Disconnected")
            st.warning("Please start the API server:\n```bash\npython api.py\n```")
        
        st.divider()
        
        # Query Settings
        st.subheader("🎛️ Query Settings")
        top_k = st.slider("Number of sources to retrieve", min_value=1, max_value=10, value=5)
        show_context = st.checkbox("Show retrieved context", value=False)
        
        st.divider()
        
        # System Stats
        st.subheader("📊 System Info")
        if st.button("Refresh Stats"):
            stats = get_stats()
            if stats:
                st.json(stats)
        
        st.divider()
        
        # About
        st.subheader("ℹ️ About")
        st.info("AI Decision Brain uses RAG (Retrieval-Augmented Generation) to answer questions based on your documents.")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Query history in session state
        if 'history' not in st.session_state:
            st.session_state.history = []
        
        # Question input
        st.subheader("💬 Ask a Question")
        question = st.text_input(
            "Enter your question:",
            placeholder="e.g., What is our Q1 2024 revenue?",
            key="question_input"
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
        
        with col_btn1:
            ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
        
        with col_btn2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_button:
            st.session_state.history = []
            st.rerun()
        
        # Process question
        if ask_button and question:
            if not is_healthy:
                st.error("⚠️ Cannot process question: API is not available")
            else:
                with st.spinner("🤔 Thinking..."):
                    success, result = query_rag(question, top_k, show_context)
                    
                    if success:
                        # Add to history
                        st.session_state.history.insert(0, {
                            'question': question,
                            'result': result,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        st.rerun()
                    else:
                        st.error(f"❌ Error: {result}")
        
        # Display history
        if st.session_state.history:
            st.divider()
            st.subheader("📜 Conversation History")
            
            for i, item in enumerate(st.session_state.history):
                with st.expander(f"Q: {item['question'][:80]}... ({item['timestamp']})", expanded=(i==0)):
                    # Question
                    st.markdown(f"**Question:**")
                    st.markdown(f'<div class="question-box">{item["question"]}</div>', unsafe_allow_html=True)
                    
                    # Answer
                    st.markdown(f"**Answer:**")
                    st.markdown(f'<div class="answer-box">{item["result"]["answer"]}</div>', unsafe_allow_html=True)
                    
                    # Sources
                    st.markdown(f"**Sources:**")
                    for j, source in enumerate(item['result']['sources'], 1):
                        similarity_pct = source['similarity'] * 100
                        st.markdown(
                            f'<div class="source-item">'
                            f'{j}. <strong>{source["filename"]}</strong> '
                            f'(Chunk {source["chunk_id"]}, Relevance: {similarity_pct:.1f}%)'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    
                    # Context (if enabled)
                    if show_context and 'context' in item['result'] and item['result']['context']:
                        with st.expander("📄 Retrieved Context"):
                            st.text_area(
                                "Context",
                                value=item['result']['context'],
                                height=200,
                                key=f"context_{i}"
                            )
                    
                    # Metadata
                    col_meta1, col_meta2 = st.columns(2)
                    with col_meta1:
                        if item['result'].get('tokens_used'):
                            st.caption(f"🔢 Tokens: {item['result']['tokens_used']}")
                    with col_meta2:
                        st.caption(f"🕐 {item['timestamp']}")
    
    with col2:
        st.subheader("💡 Example Questions")
        
        examples = [
            "What is our Q1 2024 financial performance?",
            "What are the main customer complaints?",
            "What are our competitors doing?",
            "What is our marketing strategy?",
            "What are the risk management priorities?",
            "Summarize the employee satisfaction survey"
        ]
        
        for example in examples:
            if st.button(example, key=f"example_{example}", use_container_width=True):
                st.session_state.question_input = example
                st.rerun()
        
        st.divider()
        
        st.subheader("📈 Quick Stats")
        if is_healthy and health_data:
            st.metric("Vector Store Status", health_data.get('vector_store_status', 'N/A'))
            st.metric("LLM Model", health_data.get('llm_model', 'N/A'))


if __name__ == "__main__":
    main()