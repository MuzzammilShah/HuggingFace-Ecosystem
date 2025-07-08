## Local Development Setup

### Prerequisites

- Python 3.9+
- PyTorch
- HuggingFace Transformers
- Streamlit

### Installation

1. Clone the repository
2. (Optional) Create virtual environment
3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Application

```
cd src
streamlit run app.py
```

## HuggingFace Spaces Deployment

This application can be deployed to HuggingFace Spaces:

1. Create a new Space on HuggingFace Spaces (https://huggingface.co/spaces)
2. Choose Streamlit as the SDK
3. Upload the code to the Space repository (I personally used the drag-drop method of this one as I didn't really have a complex file system)
4. The Space will automatically build and deploy the application

> **Additional Resource**: So HuggingFace has updated the way Streamlit applications can be hosted on Spaces - through Docker. [This article](https://shafiqulai.github.io/blogs/blog_4.html?id=4) was the best resource I could get which explained this new process well. Feel free to follow this through.

## Project Structure

```
├── src
│   ├── app.py                  # Main Streamlit application entry point
│   ├── nlp_engine.py           # NLP functionality implementation
│   ├── components              # UI components for each NLP task
│   │   ├── sentiment_analyzer.py
│   │   ├── text_summarizer.py
│   │   ├── entity_extractor.py
│   │   ├── question_answerer.py
│   │   ├── text_generator.py
│   │   └── semantic_search.py
│   └── utils                   # Utility functions
│       └── ui_helpers.py       # Common UI elements and formatting
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```