import streamlit as st
from utils.ui_helpers import format_sentiment_result, plot_sentiment_gauge

def show_sentiment_analyzer(nlp_engine):
    """Display the sentiment analysis UI component"""
    #st.markdown("😊😐☹️ - 📊")
    st.title("Sentiment Analysis😊😐☹️➡️📊")
    st.markdown("""
    Analyze the sentiment of text using the DistilBERT model fine-tuned on the SST-2 dataset.
    This model classifies text as either positive or negative.
    """)
    
    # Text input
    text_input = st.text_area(
        "Enter text to analyze sentiment",
        "Hugging Face is a great platform for NLP.",
        height=150
    )
    
    # Process button
    if st.button("Analyze Sentiment"):
        with st.spinner("Analyzing sentiment..."):
            # Get sentiment
            sentiment_result = nlp_engine.analyze_sentiment(text_input)
            
            # Format results
            formatted_result = format_sentiment_result(sentiment_result)
            
            # Display results
            st.markdown("### Results")
            
            # Create columns for layout
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(f"### {formatted_result['emoji']} {formatted_result['dataframe'].iloc[0]['Label']}")
                st.dataframe(formatted_result['dataframe'], hide_index=True)
            
            with col2:
                # Display gauge chart
                st.plotly_chart(
                    plot_sentiment_gauge(
                        formatted_result['score'], 
                        formatted_result['color']
                    ),
                    use_container_width=True
                )
            
    # Example section
    with st.expander("Example texts to try"):
        st.markdown("""
        - This movie was absolutely terrible. The acting was poor and the plot made no sense.
        - I had the best time at the concert! The band was amazing and the crowd was so energetic.
        - The weather today is okay, not great but not bad either.
        - I'm absolutely thrilled with the results of this project. Everything exceeded my expectations!
        """)
    
    # Information about the model
    with st.expander("About this model"):
        st.markdown("""
        **Model**: `distilbert-base-uncased-finetuned-sst-2-english`
        
        This is a DistilBERT model fine-tuned on the Stanford Sentiment Treebank v2 (SST-2) dataset.
        
        - **Accuracy**: ~91% on the SST-2 validation set
        - **Size**: 66M parameters (compared to BERT's 110M)
        - **Speed**: ~60% faster than BERT
        
        The model classifies text as either positive (LABEL_1) or negative (LABEL_0).
        """)
