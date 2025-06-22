import streamlit as st
from utils.ui_helpers import display_text_with_answer

def show_question_answerer(nlp_engine):
    """Display the question answering UI component"""
    #st.markdown("❓💬")
    st.title("Question Answering❓💬")
    st.markdown("""
    Answer questions based on a provided context using a fine-tuned RoBERTa model.
    This model extracts the answer span from the context that best answers the question.
    """)
    
    # Context input
    context = st.text_area(
        "Enter context text",
        "The capital of France is Paris. It is known for the Eiffel Tower and the Louvre Museum. Paris is located on the Seine River and is often called the City of Light. The city hosts many international events and is a major cultural and economic center in Europe.",
        height=150
    )
    
    # Question input
    question = st.text_input(
        "Enter your question",
        "What is Paris known for?"
    )
    
    # Process button
    if st.button("Answer Question"):
        if not context or not question:
            st.error("Please provide both context and question.")
        else:
            with st.spinner("Finding answer..."):
                # Get answer
                answer = nlp_engine.answer_question(question=question, context=context)
                
                # Display results
                st.markdown("### Answer")
                st.success(answer['answer'])
                
                # Display confidence score
                st.markdown(f"**Confidence Score**: {round(answer['score']*100, 2)}%")
                
                # Display answer in context
                st.markdown("### Answer in Context")
                highlighted_context = display_text_with_answer(context, answer)
                st.markdown(highlighted_context, unsafe_allow_html=True)
    
    # Example section
    with st.expander("Example context-question pairs to try"):
        st.markdown("""
        ### Example 1: Solar System
        
        **Context:**
        ```
        The Solar System is the gravitationally bound system of the Sun and the objects that orbit it. 
        The largest of these objects are the eight planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. 
        Jupiter is the largest planet, with a mass more than twice that of all the other planets combined. 
        Mars is known as the Red Planet due to its reddish appearance, which is caused by iron oxide on its surface. 
        Pluto was considered the ninth planet until 2006, when it was reclassified as a dwarf planet by the International Astronomical Union.
        ```
        
        **Questions to try:**
        - Which is the largest planet in the Solar System?
        - Why is Mars called the Red Planet?
        - How many planets are in the Solar System?
        - What happened to Pluto in 2006?
        
        ### Example 2: World War II
        
        **Context:**
        ```
        World War II was a global war that lasted from 1939 to 1945. It involved the vast majority of the world's countries forming two opposing military alliances: the Allies and the Axis. 
        The Axis powers were led by Nazi Germany, Imperial Japan, and Fascist Italy. The Allied Powers were led by Great Britain, the United States, and the Soviet Union.
        The war began on September 1, 1939, when Germany invaded Poland. Britain and France declared war on Germany two days later. 
        The United States entered the war in December 1941, following the Japanese attack on Pearl Harbor in Hawaii. 
        The war in Europe ended with Germany's surrender on May 8, 1945, while Japan surrendered on September 2, 1945, after atomic bombs were dropped on Hiroshima and Nagasaki.
        ```
        
        **Questions to try:**
        - When did World War II begin?
        - Which countries led the Axis powers?
        - Why did the United States enter the war?
        - When did Germany surrender?
        """)
    
    # Information about the model
    with st.expander("About this model"):
        st.markdown("""
        **Model**: `deepset/roberta-base-squad2`
        
        This is a RoBERTa model fine-tuned on the Stanford Question Answering Dataset (SQuAD) v2.
        
        - **Task**: Extractive question answering (finding the span of text that answers a question)
        - **Performance**: F1 score of ~80% on SQuAD v2
        - **Base Model**: RoBERTa Base (125M parameters)
        
        The model works by finding the most likely span of text in the context that answers the question. It's particularly effective for factual questions where the answer is explicitly stated in the context.
        """)
