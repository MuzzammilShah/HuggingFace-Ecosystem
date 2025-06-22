import streamlit as st

def show_text_generator(nlp_engine):
    """Display the text generation UI component"""
    #st.markdown("✍🏻🪄")
    st.title("Text Generation✍🏻🪄")
    st.markdown("""
    Generate creative text completions using GPT-2 Medium.
    This model can continue text from a prompt in a coherent and contextually relevant way.
    """)
    
    # Prompt input
    prompt = st.text_area(
        "Enter a prompt to continue",
        "In a world powered by AI,",
        height=100
    )
    
    # Parameters
    col1, col2 = st.columns(2)
    
    with col1:
        max_length = st.slider(
            "Maximum Length (tokens)",
            min_value=10,
            max_value=200,
            value=50,
            step=5,
            help="Maximum length of generated text in tokens (roughly words)"
        )
    
    with col2:
        num_sequences = st.slider(
            "Number of Completions",
            min_value=1,
            max_value=5,
            value=1,
            step=1,
            help="Number of different completions to generate"
        )
    
    # Advanced options
    with st.expander("Advanced Options"):
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=1.5,
            value=1.0,
            step=0.1,
            help="Higher values make output more random, lower values more deterministic"
        )
        
        top_p = st.slider(
            "Top-p (nucleus sampling)",
            min_value=0.1,
            max_value=1.0,
            value=0.9,
            step=0.1,
            help="Limit tokens to the most likely ones covering top_p probability mass"
        )
    
    # Process button
    if st.button("Generate Text"):
        if not prompt:
            st.error("Please provide a prompt.")
        else:
            with st.spinner("Generating text..."):
                # Get generated text
                # Note: The current NLPEngine implementation doesn't support temperature and top_p,
                # but we're adding UI controls for future enhancement
                generated_texts = nlp_engine.generate_text(
                    prompt,
                    max_length=max_length,
                    num_return_sequences=num_sequences
                )
                
                # Display results
                st.markdown("### Generated Text")
                
                for i, text in enumerate(generated_texts):
                    st.markdown(f"**Completion {i+1}:**")
                    st.markdown(f"{text['generated_text']}")
                    st.markdown("---")
    
    # Example section
    with st.expander("Example prompts to try"):
        st.markdown("""
        ### Creative Writing Prompts
        
        - Once upon a time in a forest of talking animals,
        - The abandoned spaceship contained a mysterious
        - As the detective entered the room, he immediately noticed
        - The secret to time travel was accidentally discovered by
        
        ### Business/Technical Prompts
        
        - The future of artificial intelligence will transform industries by
        - The main advantages of using cloud computing include
        - To improve customer retention, companies should focus on
        - The report outlines three key strategies for sustainable growth:
        """)
    
    # Information about the model
    with st.expander("About this model"):
        st.markdown("""
        **Model**: `gpt2-medium`
        
        GPT-2 Medium is an autoregressive language model that uses transformer architecture to generate text.
        
        - **Size**: 355M parameters (medium variant of GPT-2)
        - **Training**: Trained on a diverse dataset of internet text
        - **Capabilities**: Text completion, story generation, question answering (to some extent)
        
        The model predicts the next token based on all previous tokens in the sequence, allowing it to generate coherent and contextually relevant text continuations.
        """)
