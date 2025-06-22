import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Any, Union
import torch

def display_json_as_table(data: Union[Dict, List]):
    """Display JSON data as a formatted table"""
    if isinstance(data, list):
        # Convert list of dicts to dataframe if possible
        try:
            df = pd.DataFrame(data)
            st.dataframe(df)
        except:
            # Fall back to JSON display
            st.json(data)
    else:
        # Convert dict to dataframe with keys and values as columns
        try:
            df = pd.DataFrame(list(data.items()), columns=['Key', 'Value'])
            st.dataframe(df)
        except:
            # Fall back to JSON display
            st.json(data)

def format_sentiment_result(result: List[Dict]):
    """Format sentiment analysis result for display"""
    if not result:
        return {}
    
    # Get the first result (usually there's only one)
    item = result[0]
    
    # Create a dataframe for display
    df = pd.DataFrame({
        'Label': [item['label']],
        'Score': [round(item['score'] * 100, 2)],
    })
    
    # Add visual indicator based on sentiment
    is_positive = item['label'].lower() == 'positive'
    color = 'green' if is_positive else 'red'
    emoji = '😃' if is_positive else '😞'
    
    return {
        'dataframe': df,
        'color': color,
        'emoji': emoji,
        'score': round(item['score'] * 100, 2)
    }

def plot_sentiment_gauge(score: float, color: str):
    """Create a gauge chart for sentiment score"""
    fig = px.pie(
        values=[score, 100-score],
        names=['Score', ''],
        color_discrete_sequence=[color, '#F0F2F6'],
        hole=0.7,
    )
    
    fig.update_layout(
        annotations=[dict(text=f"{score}%", x=0.5, y=0.5, font_size=20, showarrow=False)],
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        height=200,
    )
    
    return fig

def format_ner_results(ner_results: List[Dict]):
    """Format NER results for display"""
    # Convert to DataFrame for easier display
    if not ner_results:
        return pd.DataFrame()
    
    entities_data = []
    for entity in ner_results:
        entities_data.append({
            'Entity': entity['word'],
            'Type': entity['entity_group'],
            'Score': round(entity['score'] * 100, 2),
            'Start': entity['start'],
            'End': entity['end']
        })
    
    return pd.DataFrame(entities_data)

def highlight_entities_in_text(text: str, entities: List[Dict]):
    """Highlight entities in text for display"""
    if not entities:
        return text
    
    # Sort entities by start position in reverse order
    # to avoid index shifting when replacing
    sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
    
    # Convert text to HTML with highlights
    html_text = text
    for entity in sorted_entities:
        start = entity['start']
        end = entity['end']
        entity_text = text[start:end]
        entity_type = entity['entity_group']
        
        # Define color based on entity type
        color_map = {
            'PER': '#FFD700',  # Person - Gold
            'ORG': '#98FB98',  # Organization - Pale Green
            'LOC': '#ADD8E6',  # Location - Light Blue
            'MISC': '#FFA07A'   # Miscellaneous - Light Salmon
        }
        
        color = color_map.get(entity_type, '#D3D3D3')  # Default to light gray
        
        # Replace text with highlighted version
        highlight = f'<span style="background-color: {color}; padding: 0px 2px; border-radius: 3px;" title="{entity_type} ({round(entity["score"]*100)}%)">{entity_text}</span>'
        html_text = html_text[:start] + highlight + html_text[end:]
    
    return html_text

def plot_similarity_heatmap(query: str, texts: List[str], similarities: List[float]):
    """Create a bar chart for similarity scores"""
    # Create a DataFrame with the data
    df = pd.DataFrame({
        'Text': texts,
        'Similarity': similarities
    })
    
    # Sort by similarity score in descending order
    df = df.sort_values('Similarity', ascending=False)
    
    # Create the bar chart
    fig = px.bar(
        df,
        x='Similarity',
        y='Text',
        orientation='h',
        title=f'Similarity to: "{query}"',
        labels={'Similarity': 'Cosine Similarity', 'Text': 'Corpus Text'},
        range_x=[0, 1]
    )
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    return fig

def display_text_with_answer(context: str, answer: Dict):
    """Display context text with highlighted answer"""
    if not answer or 'start' not in answer or 'end' not in answer:
        return context
    
    start = answer['start']
    end = answer['end']
    answer_text = answer['answer']
    
    # Replace the answer with a highlighted version
    highlighted_text = (
        context[:start] +
        f'<span style="background-color: #FFFF00; font-weight: bold;">{answer_text}</span>' +
        context[end:]
    )
    
    return highlighted_text
