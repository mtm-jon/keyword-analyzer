import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
import json
import re
from scipy import stats

# Page configuration
st.set_page_config(page_title="SEO Content Analyzer", layout="wide")

# Initialize session state
if 'embeddings_cache' not in st.session_state:
    st.session_state['embeddings_cache'] = {}

def get_embedding(text: str) -> List[float]:
    """Get embedding from local server"""
    try:
        response = requests.post(
            "http://localhost:5001/embed",
            json={
                "text": text,
                "task": "RETRIEVAL_DOCUMENT"
            }
        )
        if response.status_code == 200:
            return response.json()["embedding"]
        else:
            st.error(f"Error from embedding server: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to embedding server: {str(e)}")
        return None

def chunk_text(text: str, chunk_size: int = 200) -> List[str]:
    """Split text into chunks of approximately equal size"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_length += len(word) + 1  # +1 for space
        if current_length > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def calculate_chunk_similarity(chunks: List[str]) -> pd.DataFrame:
    """Calculate similarity between consecutive chunks"""
    chunk_embeddings = []
    for chunk in chunks:
        if chunk in st.session_state['embeddings_cache']:
            embedding = st.session_state['embeddings_cache'][chunk]
        else:
            embedding = get_embedding(chunk)
            st.session_state['embeddings_cache'][chunk] = embedding
        chunk_embeddings.append(embedding)
    
    similarities = []
    for i in range(len(chunk_embeddings)-1):
        similarity = cosine_similarity(
            [chunk_embeddings[i]], 
            [chunk_embeddings[i+1]]
        )[0][0]
        similarities.append(similarity)
    
    return pd.DataFrame({
        'Chunk Pair': [f"Chunks {i+1}-{i+2}" for i in range(len(similarities))],
        'Similarity': similarities
    })

def analyze_topic_diversity(chunks: List[str]) -> Dict[str, float]:
    """Analyze topic diversity across content chunks"""
    chunk_embeddings = []
    for chunk in chunks:
        if chunk in st.session_state['embeddings_cache']:
            embedding = st.session_state['embeddings_cache'][chunk]
        else:
            embedding = get_embedding(chunk)
            st.session_state['embeddings_cache'][chunk] = embedding
        chunk_embeddings.append(embedding)
    
    # Calculate pairwise similarities
    similarity_matrix = cosine_similarity(chunk_embeddings)
    
    # Calculate metrics
    avg_similarity = np.mean(similarity_matrix[np.triu_indices(len(similarity_matrix), k=1)])
    std_similarity = np.std(similarity_matrix[np.triu_indices(len(similarity_matrix), k=1)])
    
    return {
        'average_similarity': avg_similarity,
        'similarity_std': std_similarity,
        'diversity_score': 1 - avg_similarity
    }

def identify_serp_features(text: str) -> Dict[str, bool]:
    """Identify potential SERP feature opportunities"""
    features = {
        'featured_snippet': False,
        'people_also_ask': False,
        'list_format': False,
        'table_format': False,
        'how_to': False
    }
    
    # Check for list formats
    if re.search(r'\n\s*[\-\*\d+]\s+', text):
        features['list_format'] = True
    
    # Check for table-like content
    if '|' in text or '\t' in text:
        features['table_format'] = True
    
    # Check for how-to content
    if re.search(r'how\s+to|steps?|guide', text.lower()):
        features['how_to'] = True
    
    # Check for Q&A format (People Also Ask potential)
    if re.search(r'\?.*\n', text):
        features['people_also_ask'] = True
    
    # Check for featured snippet potential
    if features['list_format'] or features['table_format'] or features['how_to']:
        features['featured_snippet'] = True
    
    return features

def analyze_competitors(main_text: str, competitor_texts: List[str]) -> Dict[str, Any]:
    """Analyze content gaps and similarities with competitors"""
    # Get embeddings
    main_embedding = get_embedding(main_text)
    competitor_embeddings = []
    
    for text in competitor_texts:
        if text in st.session_state['embeddings_cache']:
            embedding = st.session_state['embeddings_cache'][text]
        else:
            embedding = get_embedding(text)
            st.session_state['embeddings_cache'][text] = embedding
        competitor_embeddings.append(embedding)
    
    # Calculate similarities
    similarities = []
    for comp_emb in competitor_embeddings:
        similarity = cosine_similarity([main_embedding], [comp_emb])[0][0]
        similarities.append(similarity)
    
    # Analyze content gaps
    avg_competitor_embedding = np.mean(competitor_embeddings, axis=0)
    gap_score = 1 - cosine_similarity([main_embedding], [avg_competitor_embedding])[0][0]
    
    return {
        'competitor_similarities': similarities,
        'average_similarity': np.mean(similarities),
        'content_gap_score': gap_score
    }

def detect_anomalies(similarities: List[float], threshold: float = 2) -> List[int]:
    """Detect anomalies in content similarity scores"""
    z_scores = stats.zscore(similarities)
    return [i for i, z in enumerate(z_scores) if abs(z) > threshold]

def main():
    st.title("SEO Content Analyzer with Vector Embeddings")
    
    # Input section
    st.header("Content Input")
    main_content = st.text_area("Enter your main content", height=200)
    
    # Competitor analysis section
    st.header("Competitor Analysis")
    num_competitors = st.number_input("Number of competitor content to analyze", min_value=0, max_value=5, value=0)
    competitor_contents = []
    
    if num_competitors > 0:
        for i in range(num_competitors):
            competitor_content = st.text_area(f"Competitor Content {i+1}", height=150)
            competitor_contents.append(competitor_content)
    
    if st.button("Analyze Content"):
        if main_content:
            # Progress bar
            progress_bar = st.progress(0)
            
            # 1. Chunk Analysis
            st.subheader("Content Structure Analysis")
            chunks = chunk_text(main_content)
            chunk_similarities = calculate_chunk_similarity(chunks)
            
            # Plot chunk similarities
            fig_chunks = px.line(chunk_similarities, x='Chunk Pair', y='Similarity',
                               title='Content Flow Analysis')
            st.plotly_chart(fig_chunks)
            
            progress_bar.progress(20)
            
            # 2. Topic Diversity Analysis
            st.subheader("Topic Diversity Analysis")
            diversity_metrics = analyze_topic_diversity(chunks)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Average Topic Similarity", f"{diversity_metrics['average_similarity']:.2f}")
            col2.metric("Topic Variance", f"{diversity_metrics['similarity_std']:.2f}")
            col3.metric("Diversity Score", f"{diversity_metrics['diversity_score']:.2f}")
            
            progress_bar.progress(40)
            
            # 3. SERP Features Analysis
            st.subheader("SERP Feature Opportunities")
            serp_features = identify_serp_features(main_content)
            
            for feature, present in serp_features.items():
                st.checkbox(
                    feature.replace('_', ' ').title(),
                    value=present,
                    disabled=True
                )
            
            progress_bar.progress(60)
            
            # 4. Competitor Analysis
            if competitor_contents and all(competitor_contents):
                st.subheader("Competitor Analysis")
                competitor_analysis = analyze_competitors(main_content, competitor_contents)
                
                # Plot competitor similarities
                fig_competitors = go.Figure()
                fig_competitors.add_trace(go.Bar(
                    x=[f"Competitor {i+1}" for i in range(len(competitor_analysis['competitor_similarities']))],
                    y=competitor_analysis['competitor_similarities'],
                    name="Similarity Score"
                ))
                fig_competitors.update_layout(title="Content Similarity with Competitors")
                st.plotly_chart(fig_competitors)
                
                st.metric("Content Gap Score", f"{competitor_analysis['content_gap_score']:.2f}")
                
                # Detect anomalies
                anomalies = detect_anomalies(competitor_analysis['competitor_similarities'])
                if anomalies:
                    st.warning("Potential content gaps detected with competitors: " + 
                             ", ".join([f"Competitor {i+1}" for i in anomalies]))
            
            progress_bar.progress(80)
            
            # 5. Recommendations
            st.subheader("SEO Recommendations")
            recommendations = []
            
            # Content structure recommendations
            if any(sim < 0.5 for sim in chunk_similarities['Similarity']):
                recommendations.append("⚠️ Consider improving content flow between chunks with low similarity scores")
            
            # Topic diversity recommendations
            if diversity_metrics['diversity_score'] < 0.3:
                recommendations.append("📝 Content might be too focused - consider expanding topic coverage")
            elif diversity_metrics['diversity_score'] > 0.7:
                recommendations.append("🎯 Content might be too diverse - consider tightening topic focus")
            
            # SERP feature recommendations
            for feature, present in serp_features.items():
                if not present:
                    recommendations.append(f"💡 Consider adding {feature.replace('_', ' ')} content for SERP features")
            
            # Competitor-based recommendations
            if 'competitor_analysis' in locals():
                if competitor_analysis['content_gap_score'] > 0.3:
                    recommendations.append("🔍 Significant content gaps detected - review competitor topics")
            
            for rec in recommendations:
                st.write(rec)
            
            progress_bar.progress(100)
            
        else:
            st.error("Please enter some content to analyze")

if __name__ == "__main__":
    main()
