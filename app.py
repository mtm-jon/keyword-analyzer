import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
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

def analyze_keyword_relevance(content: str, keywords: List[str]) -> pd.DataFrame:
    """Analyze content relevance to multiple keywords"""
    content_embedding = get_embedding(content)
    if not content_embedding:
        return pd.DataFrame()
    
    results = []
    for keyword in keywords:
        keyword_embedding = get_embedding(keyword)
        if keyword_embedding:
            similarity = cosine_similarity([content_embedding], [keyword_embedding])[0][0]
            results.append({
                "Keyword": keyword,
                "Similarity Score (%)": round(similarity * 100, 2)
            })
    
    return pd.DataFrame(results).sort_values("Similarity Score (%)", ascending=False)

def analyze_chunk_keyword_relevance(chunks: List[str], keywords: List[str]) -> Dict[str, List[float]]:
    """Analyze each chunk's relevance to keywords"""
    chunk_keyword_scores = {keyword: [] for keyword in keywords}
    
    for chunk in chunks:
        chunk_embedding = get_embedding(chunk)
        if chunk_embedding:
            for keyword in keywords:
                keyword_embedding = get_embedding(keyword)
                if keyword_embedding:
                    similarity = cosine_similarity([chunk_embedding], [keyword_embedding])[0][0]
                    chunk_keyword_scores[keyword].append(similarity * 100)
    
    return chunk_keyword_scores

def calculate_chunk_similarity(chunks: List[str]) -> pd.DataFrame:
    """Calculate similarity between consecutive chunks"""
    chunk_embeddings = []
    for chunk in chunks:
        if chunk in st.session_state['embeddings_cache']:
            embedding = st.session_state['embeddings_cache'][chunk]
        else:
            embedding = get_embedding(chunk)
            if embedding:
                st.session_state['embeddings_cache'][chunk] = embedding
        if embedding:
            chunk_embeddings.append(embedding)
    
    similarities = []
    for i in range(len(chunk_embeddings)-1):
        if chunk_embeddings[i] is not None and chunk_embeddings[i+1] is not None:
            similarity = cosine_similarity(
                [chunk_embeddings[i]], 
                [chunk_embeddings[i+1]]
            )[0][0]
            similarities.append(similarity)
        else:
            similarities.append(0.0)
    
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
            if embedding:
                st.session_state['embeddings_cache'][chunk] = embedding
        if embedding:
            chunk_embeddings.append(embedding)
    
    if not chunk_embeddings:
        return {
            'average_similarity': 0.0,
            'similarity_std': 0.0,
            'diversity_score': 0.0
        }
    
    similarity_matrix = cosine_similarity(chunk_embeddings)
    avg_similarity = np.mean(similarity_matrix[np.triu_indices(len(similarity_matrix), k=1)])
    std_similarity = np.std(similarity_matrix[np.triu_indices(len(similarity_matrix), k=1)])
    
    return {
        'average_similarity': avg_similarity,
        'similarity_std': std_similarity,
        'diversity_score': 1 - avg_similarity
    }

def main():
    st.title("🎯 SEO Content & Keyword Analyzer")
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        main_content = st.text_area("Enter your content", height=200)
    
    with col2:
        keywords_input = st.text_area(
            "Enter target keywords (one per line)",
            height=200,
            placeholder="keyword 1\nkeyword 2\nkeyword 3"
        )
    
    if st.button("Analyze Content", type="primary"):
        if not main_content or not keywords_input:
            st.error("Please provide both content and keywords")
            return
        
        keywords = [k.strip() for k in keywords_input.split('\n') if k.strip()]
        
        try:
            # Progress bar
            progress_bar = st.progress(0)
            
            # Create tabs for different analyses
            tab1, tab2, tab3 = st.tabs(["Keyword Analysis", "Content Structure", "Topic Analysis"])
            
            with tab1:
                # Overall keyword relevance
                st.subheader("Overall Keyword Relevance")
                keyword_relevance = analyze_keyword_relevance(main_content, keywords)
                
                if not keyword_relevance.empty:
                    fig = px.bar(
                        keyword_relevance,
                        x="Keyword",
                        y="Similarity Score (%)",
                        title="Keyword Semantic Alignment",
                        color="Similarity Score (%)",
                        color_continuous_scale="viridis"
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Chunk-level keyword analysis
                st.subheader("Content Section Keyword Relevance")
                chunks = chunk_text(main_content)
                chunk_keyword_scores = analyze_chunk_keyword_relevance(chunks, keywords)
                
                # Plot chunk-keyword heatmap
                if chunk_keyword_scores and any(scores for scores in chunk_keyword_scores.values()):
                    fig_heatmap = go.Figure(data=go.Heatmap(
                        z=list(chunk_keyword_scores.values()),
                        x=[f"Chunk {i+1}" for i in range(len(next(iter(chunk_keyword_scores.values()))))],
                        y=list(chunk_keyword_scores.keys()),
                        colorscale="Viridis",
                        colorbar=dict(title="Relevance Score (%)")
                    ))
                    fig_heatmap.update_layout(title="Keyword Relevance by Content Section")
                    st.plotly_chart(fig_heatmap, use_container_width=True)
            
            progress_bar.progress(33)
            
            with tab2:
                st.subheader("Content Structure Analysis")
                chunks = chunk_text(main_content)
                chunk_similarities = calculate_chunk_similarity(chunks)
                
                if not chunk_similarities.empty:
                    fig_chunks = px.line(
                        chunk_similarities,
                        x='Chunk Pair',
                        y='Similarity',
                        title='Content Flow Analysis'
                    )
                    st.plotly_chart(fig_chunks, use_container_width=True)
                
                # Basic metrics
                words = main_content.split()
                paragraphs = main_content.split('\n\n')
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Words", len(words))
                col2.metric("Total Paragraphs", len(paragraphs))
                col3.metric("Avg Words per Paragraph", round(len(words)/len(paragraphs) if paragraphs else 0))
            
            progress_bar.progress(66)
            
            with tab3:
                st.subheader("Topic Analysis")
                diversity_metrics = analyze_topic_diversity(chunks)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Topic Consistency", f"{diversity_metrics['average_similarity']:.2%}")
                col2.metric("Topic Variance", f"{diversity_metrics['similarity_std']:.2%}")
                col3.metric("Topic Diversity", f"{diversity_metrics['diversity_score']:.2%}")
                
                # Recommendations based on analysis
                st.subheader("Content Recommendations")
                recommendations = []
                
                # Keyword recommendations
                if not keyword_relevance.empty:
                    best_keyword = keyword_relevance.iloc[0]
                    if best_keyword["Similarity Score (%)"] > 80:
                        recommendations.append(f"✅ Strong alignment with keyword: '{best_keyword['Keyword']}'")
                    elif best_keyword["Similarity Score (%)"] > 60:
                        recommendations.append(f"📈 Moderate alignment with keyword: '{best_keyword['Keyword']}'. Consider strengthening the focus.")
                    else:
                        recommendations.append(f"⚠️ Low alignment with all keywords. Consider revising content to better target your keywords.")
                
                # Topic diversity recommendations
                if diversity_metrics['diversity_score'] < 0.3:
                    recommendations.append("📝 Content might be too focused - consider expanding topic coverage")
                elif diversity_metrics['diversity_score'] > 0.7:
                    recommendations.append("🎯 Content might be too diverse - consider tightening topic focus")
                
                # Content structure recommendations
                if any(sim < 0.5 for sim in chunk_similarities['Similarity']):
                    recommendations.append("⚠️ Some content sections have low topical connection - consider improving content flow")
                
                for rec in recommendations:
                    st.write(rec)
            
            progress_bar.progress(100)
            
            # Export options
            st.sidebar.subheader("Export Results")
            if not keyword_relevance.empty:
                csv = keyword_relevance.to_csv(index=False)
                st.sidebar.download_button(
                    label="Download Analysis as CSV",
                    data=csv,
                    file_name="seo_analysis_results.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")

if __name__ == "__main__":
    main()
