import streamlit as st
from utils.ui_helpers import format_ner_results, highlight_entities_in_text

def show_entity_extractor(nlp_engine):
    """Display the named entity recognition UI component"""
    #st.markdown("🏷️")
    st.title("Named Entity Recognition🏷️")
    st.markdown("""
    Extract named entities from text using a fine-tuned BERT model.
    This model can identify entities such as persons, organizations, locations, and more.
    """)
    
    # Text input
    text_input = st.text_area(
        "Enter text to extract entities",
        "Apple Inc. is looking at buying U.K. startup for $1 billion. Tim Cook is the CEO. The meeting is in New York.",
        height=150
    )
    
    # Process button
    if st.button("Extract Entities"):
        with st.spinner("Extracting entities..."):
            # Get entities
            entities = nlp_engine.extract_entities(text_input)
            
            # Display results
            st.markdown("### Results")
            
            # Show highlighted text
            st.markdown("#### Text with highlighted entities")
            highlighted_html = highlight_entities_in_text(text_input, entities)
            st.markdown(highlighted_html, unsafe_allow_html=True)
            
            # Show entities table
            st.markdown("#### Extracted entities")
            entities_df = format_ner_results(entities)
            st.dataframe(entities_df, hide_index=True)
            
            # Display entity type legend
            st.markdown("#### Entity Types")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<span style="background-color: #FFD700; padding: 0px 5px;">PER</span> - Person', unsafe_allow_html=True)
            with col2:
                st.markdown('<span style="background-color: #98FB98; padding: 0px 5px;">ORG</span> - Organization', unsafe_allow_html=True)
            with col3:
                st.markdown('<span style="background-color: #ADD8E6; padding: 0px 5px;">LOC</span> - Location', unsafe_allow_html=True)
            with col4:
                st.markdown('<span style="background-color: #FFA07A; padding: 0px 5px;">MISC</span> - Miscellaneous', unsafe_allow_html=True)
    
    # Example section
    with st.expander("Example texts to try"):
        st.markdown("""
        ### Example 1: Business News
        
        ```
        Microsoft Corporation announced a new partnership with OpenAI yesterday. Satya Nadella, CEO of Microsoft, stated that this collaboration would accelerate AI research. The deal, worth approximately $10 billion, was finalized in Seattle last week. Google and Amazon are reportedly watching this development closely.
        ```
        
        ### Example 2: Sports News
        
        ```
        The Los Angeles Lakers defeated the Boston Celtics 114-106 at Staples Center on Saturday. LeBron James scored 32 points, while Jayson Tatum led the Celtics with 28 points. The Lakers will next face the Golden State Warriors in San Francisco on Tuesday.
        ```
        
        ### Example 3: Historical Context
        
        ```
        The Treaty of Versailles was signed in France on June 28, 1919, exactly five years after the assassination of Archduke Franz Ferdinand in Sarajevo. President Woodrow Wilson represented the United States at the Paris Peace Conference. Germany was forced to accept terms that severely limited its military capabilities.
        ```
        """)
    
    # Information about the model
    with st.expander("About this model"):
        st.markdown("""
        **Model**: `dslim/bert-base-NER`
        
        This is a fine-tuned BERT model for Named Entity Recognition (NER) trained on the CoNLL-2003 dataset.
        
        - **Entities**: Recognizes PER (Person), ORG (Organization), LOC (Location), and MISC (Miscellaneous)
        - **Performance**: F1 Score of ~91% on CoNLL-2003 test set
        - **Base Model**: BERT Base
        
        The model uses BIO tagging scheme (Beginning, Inside, Outside) to mark entity spans and supports aggregation for better entity grouping.
        """)
