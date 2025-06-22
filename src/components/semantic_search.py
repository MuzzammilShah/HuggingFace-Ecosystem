import streamlit as st
import torch
from utils.ui_helpers import plot_similarity_heatmap

def show_semantic_search(nlp_engine):
    """Display the semantic search UI component"""
    #st.markdown("🔍🧠")
    st.title("Semantic Search🔍🧠")
    st.markdown("""
    Search for semantically similar texts using sentence embeddings.
    This tool converts text to vector representations and computes similarity.
    """)
    
    # Create tabs for different modes
    tab1, tab2 = st.tabs(["Semantic Search", "Text Similarity"])
    
    # TAB 1: Semantic Search
    with tab1:
        st.markdown("### Search in a Corpus of Texts")
        
        # Corpus input
        corpus_text = st.text_area(
            "Enter corpus texts (one text per line)",
            "The weather is sunny today.\nI enjoy walking in the park on a beautiful day.\nAI is transforming many industries.",
            height=150,
            help="Enter multiple sentences or paragraphs, one per line"
        )
        
        # Convert to list
        corpus = [text.strip() for text in corpus_text.split('\n') if text.strip()]
        
        # Query input
        query = st.text_input(
            "Enter your search query",
            "What is the forecast for today?"
        )
        
        # Process button
        if st.button("Search", key="search_button"):
            if not corpus or not query:
                st.error("Please provide both corpus texts and a search query.")
            else:
                with st.spinner("Computing similarities..."):
                    # Get embeddings
                    query_embedding = nlp_engine.get_embeddings(query)
                    corpus_embeddings = nlp_engine.get_embeddings(corpus)
                    
                    # Ensure query embedding is 2D for comparison
                    if query_embedding.ndim == 1:
                        query_embedding = query_embedding.unsqueeze(0)
                    
                    # Compute similarities
                    similarities = torch.nn.functional.cosine_similarity(
                        query_embedding, 
                        corpus_embeddings, 
                        dim=1
                    ).tolist()
                    
                    # Display results
                    st.markdown("### Search Results")
                    
                    # Plot similarities
                    fig = plot_similarity_heatmap(query, corpus, similarities)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show sorted results in a table
                    results = []
                    for i, (text, score) in enumerate(sorted(zip(corpus, similarities), key=lambda x: x[1], reverse=True)):
                        results.append({
                            "Rank": i + 1,
                            "Text": text,
                            "Similarity": f"{score:.4f}"
                        })
                    
                    st.table(results)
    
    # TAB 2: Text Similarity
    with tab2:
        st.markdown("### Compare Two Texts")
        
        # Text inputs
        col1, col2 = st.columns(2)
        
        with col1:
            text1 = st.text_area(
                "Text 1",
                "The weather forecast predicts rain tomorrow.",
                height=150
            )
        
        with col2:
            text2 = st.text_area(
                "Text 2",
                "According to meteorologists, precipitation is expected the following day.",
                height=150
            )
        
        # Process button
        if st.button("Compare", key="compare_button"):
            if not text1 or not text2:
                st.error("Please provide both texts to compare.")
            else:
                with st.spinner("Computing similarity..."):
                    # Get embeddings
                    embedding1 = nlp_engine.get_embeddings(text1)
                    embedding2 = nlp_engine.get_embeddings(text2)
                    
                    # Ensure embeddings are 2D for comparison
                    if embedding1.ndim == 1:
                        embedding1 = embedding1.unsqueeze(0)
                    if embedding2.ndim == 1:
                        embedding2 = embedding2.unsqueeze(0)
                    
                    # Compute similarity
                    similarity = torch.nn.functional.cosine_similarity(
                        embedding1, 
                        embedding2, 
                        dim=1
                    ).item()
                    
                    # Display result
                    st.markdown("### Similarity Result")
                    
                    # Create a visual representation of similarity
                    similarity_percentage = round(similarity * 100, 2)
                    st.progress(similarity)
                    
                    # Show the similarity score
                    if similarity_percentage > 80:
                        st.success(f"Similarity Score: {similarity_percentage}% (Very Similar)")
                    elif similarity_percentage > 60:
                        st.info(f"Similarity Score: {similarity_percentage}% (Moderately Similar)")
                    elif similarity_percentage > 40:
                        st.warning(f"Similarity Score: {similarity_percentage}% (Somewhat Similar)")
                    else:
                        st.error(f"Similarity Score: {similarity_percentage}% (Not Very Similar)")
    
    # Example section
    with st.expander("Example texts to try"):
        st.markdown("""
        ### Example Corpus for Semantic Search
        
        ```
        Artificial intelligence is revolutionizing healthcare through early disease detection.
        Machine learning algorithms can predict patient outcomes based on historical data.
        The automotive industry is investing heavily in self-driving car technology.
        Climate change is causing more frequent and severe weather events globally.
        Regular exercise and a balanced diet are essential for maintaining good health.
        The global economy faces significant challenges due to supply chain disruptions.
        Renewable energy sources are becoming increasingly cost-competitive with fossil fuels.
        ```
        
        **Example Queries:**
        - How is AI helping doctors?
        - What's happening with autonomous vehicles?
        - How can I stay healthy?
        - What are the economic trends currently?
        
        ### Example Text Pairs for Similarity Comparison
        
        **Similar Pairs:**
        
        Text 1: `The film received positive reviews from critics and performed well at the box office.`
        Text 2: `The movie was praised by reviewers and was commercially successful.`
        
        Text 1: `The company announced a significant increase in quarterly earnings.`
        Text 2: `The firm reported a substantial growth in profits for the last quarter.`
        
        **Dissimilar Pairs:**
        
        Text 1: `The recipe calls for two tablespoons of olive oil and fresh herbs.`
        Text 2: `The basketball game went into overtime after a tied score at the final buzzer.`
        
        Text 1: `Quantum computing leverages principles of quantum mechanics to process information.`
        Text 2: `The annual flower festival attracts thousands of visitors to the botanical gardens.`
        """)
    
    # Information about the model
    with st.expander("About this model"):
        st.markdown("""
        **Model**: `sentence-transformers/all-MiniLM-L6-v2`
        
        This is a sentence transformer model that generates fixed-size embeddings for text.
        
        - **Size**: 80M parameters (smaller than BERT Base)
        - **Embedding Dimension**: 384
        - **Performance**: 58.80% accuracy on Semantic Textual Similarity (STS) benchmark
        
        The model maps sentences or paragraphs to a dense vector space where semantically similar texts are close to each other, enabling semantic search and similarity comparisons.
        """)
