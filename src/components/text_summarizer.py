import streamlit as st

def show_text_summarizer(nlp_engine):
    """Display the text summarization UI component"""
    #st.markdown("📄➡️📎")
    st.title("Text Summarization📄➡️📎")
    st.markdown("""
    Generate concise summaries of longer texts using the BART-Large-CNN model.
    This model is fine-tuned on CNN Daily Mail, a dataset of news articles paired with summaries.
    """)
    
    # Text input
    text_input = st.text_area(
        "Enter text to summarize",
        """The Hugging Face ecosystem provides a wide array of tools and models for natural language processing.
It includes transformers for state-of-the-art models, datasets for accessing and sharing data,
and a model hub for discovering and using pre-trained models. Developers can leverage these
resources to build powerful NLP applications with relative ease. The platform also supports
various tasks such as text classification, summarization, translation, and question answering.
The quick brown fox jumps over the lazy dog. This sentence is repeated multiple times to ensure
the text is long enough for summarization to be meaningful. The quick brown fox jumps over the lazy dog.
The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.""",
        height=250
    )
    
    # Parameters
    col1, col2 = st.columns(2)
    
    with col1:
        min_length = st.slider(
            "Minimum Length (words)",
            min_value=10,
            max_value=100,
            value=30,
            step=5,
            help="Minimum length of the summary in words"
        )
    
    with col2:
        max_length = st.slider(
            "Maximum Length (words)",
            min_value=50,
            max_value=500,
            value=150,
            step=10,
            help="Maximum length of the summary in words"
        )
    
    # Process button
    if st.button("Generate Summary"):
        if len(text_input.split()) < min_length:
            st.error(f"Input text is too short. It should have at least {min_length} words.")
        else:
            with st.spinner("Generating summary..."):
                # Get summary
                summary_result = nlp_engine.summarize_text(
                    text_input,
                    max_length=max_length,
                    min_length=min_length
                )
                
                # Display results
                st.markdown("### Summary")
                st.info(summary_result[0]['summary_text'])
                
                # Display statistics
                input_word_count = len(text_input.split())
                summary_word_count = len(summary_result[0]['summary_text'].split())
                reduction = round((1 - summary_word_count / input_word_count) * 100, 1)
                
                st.markdown(f"""
                **Statistics:**
                - Original text: {input_word_count} words
                - Summary: {summary_word_count} words
                - Reduction: {reduction}%
                """)
    
    # Example section
    with st.expander("Example texts to try"):
        st.markdown("""
        ### Example 1: Scientific Article
        
        ```
        Recent advances in artificial intelligence have led to significant breakthroughs in natural language processing. 
        Transformer models like BERT, GPT, and T5 have demonstrated remarkable capabilities in understanding and generating human language. 
        These models leverage self-attention mechanisms to process sequences of text in parallel, capturing long-range dependencies more effectively than previous architectures like RNNs or LSTMs. 
        Pre-training on vast corpora of text allows these models to learn general language representations that can be fine-tuned for specific downstream tasks with relatively small amounts of labeled data. 
        Applications of these technologies include machine translation, text summarization, question answering, and sentiment analysis. 
        Despite their impressive performance, challenges remain in areas such as computational efficiency, interpretability, and ethical considerations regarding bias and fairness. 
        Researchers continue to explore methods for reducing model size while maintaining performance, as well as techniques for making models more transparent and accountable.
        ```
        
        ### Example 2: News Article
        
        ```
        The city council voted yesterday to approve the controversial downtown development project, following a heated debate that lasted nearly five hours. 
        The $500 million project will include a 40-story residential tower, 100,000 square feet of retail space, and a public park. 
        Supporters argue that the development will create jobs and revitalize the downtown area, which has struggled economically in recent years. 
        They point to estimates suggesting the project will generate 1,500 construction jobs and 800 permanent positions once completed.
        However, opponents raised concerns about increased traffic, potential environmental impacts, and the displacement of existing small businesses in the area. 
        Community activist groups held protests outside city hall, with signs reading "People Over Profit" and "Save Our Neighborhood."
        The final vote was 7-4 in favor of the project, with councilmembers from the downtown districts voting against it. 
        Mayor Johnson, who has championed the development since its proposal two years ago, called the decision "a crucial step forward for our city's future."
        Construction is expected to begin next spring and last approximately three years. 
        The developer has agreed to include 15% affordable housing units and contribute $5 million to a community benefits fund as part of the approval conditions.
        ```
        """)
    
    # Information about the model
    with st.expander("About this model"):
        st.markdown("""
        **Model**: `facebook/bart-large-cnn`
        
        BART (Bidirectional and Auto-Regressive Transformers) is a transformer encoder-decoder model fine-tuned on CNN Daily Mail, a large dataset of news articles paired with summaries.
        
        - **Size**: 400M parameters
        - **Training**: Pre-trained with a denoising objective on a large text corpus, then fine-tuned on CNN/DM dataset
        - **Performance**: State-of-the-art results on various summarization benchmarks
        
        This model is particularly effective at generating concise, coherent summaries that capture the main points of news articles and other informative texts.
        """)
