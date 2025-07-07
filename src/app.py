import streamlit as st
import torch
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
from components.sentiment_analyzer import show_sentiment_analyzer
from components.text_summarizer import show_text_summarizer
from components.entity_extractor import show_entity_extractor
from components.question_answerer import show_question_answerer
from components.text_generator import show_text_generator
from components.semantic_search import show_semantic_search

# Import NLP Engine
from nlp_engine import NLPEngine

# Set page config
st.set_page_config(
    page_title="HuggingFace Ecosystem",
    page_icon="🤗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the NLP Engine initialization
@st.cache_resource
def get_nlp_engine():
    with st.spinner('Loading NLP models... This might take a while.'):
        # Check for available hardware acceleration
        if torch.cuda.is_available():
            st.sidebar.success("CUDA GPU detected! Using GPU acceleration.")
            device = 0  # CUDA device
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            st.sidebar.success("Apple MPS detected! Using MPS acceleration.")
            device = 'mps'  # MPS device
        else:
            st.sidebar.info("No GPU detected. Using CPU.")
            device = -1  # CPU
        
        return NLPEngine(device=device)

def main():
    # Initialize NLP Engine
    nlp_engine = get_nlp_engine()
    
    # Sidebar
    st.sidebar.title("🤗 HuggingFace Ecosystem - NLP Playground")
    st.sidebar.markdown("---")
    
    # Model selection
    task = st.sidebar.selectbox(
        "Choose NLP Task",
        [
            "Sentiment Analysis",
            "Text Summarization",
            "Named Entity Recognition",
            "Question Answering",
            "Text Generation",
            "Semantic Search"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    This application demonstrates various NLP capabilities using HuggingFace's Transformers library.
    
    ### Instructions
    1. Select an NLP task from the dropdown menu
    2. Enter the required inputs
    3. Adjust parameters if needed
    4. Run the model and view results
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(
    "Created by [Muhammed Shah](https://muhammedshah.com) with ❤️ using HuggingFace and Streamlit.<br>",
    unsafe_allow_html=True
    )
    
    # Main content
    if task == "Sentiment Analysis":
        show_sentiment_analyzer(nlp_engine)
    elif task == "Text Summarization":
        show_text_summarizer(nlp_engine)
    elif task == "Named Entity Recognition":
        show_entity_extractor(nlp_engine)
    elif task == "Question Answering":
        show_question_answerer(nlp_engine)
    elif task == "Text Generation":
        show_text_generator(nlp_engine)
    elif task == "Semantic Search":
        show_semantic_search(nlp_engine)

if __name__ == "__main__":
    main()
