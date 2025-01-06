import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import re

# Configuration
EMBEDDING_SERVER_URL = "http://localhost:5001"

def get_embedding(text):
    """Get embedding from local server"""
    try:
        response = requests.post(
            f"{EMBEDDING_SERVER_URL}/embed",
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

def cosine_similarity(vec_a, vec_b):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot_product / (norm_a * norm_b)

# Set up the Streamlit interface
st.title("SEO Content Analyzer")
st.markdown("""
This tool analyzes your content using semantic embeddings to measure alignment with target keywords.
Make sure your embedding server is running locally before using this tool.
""")

# Create input areas
col1, col2 = st.columns([2, 1])

with col1:
    content = st.text_area(
        "Your Content",
        height=200,
        placeholder="Paste your content here..."
    )

with col2:
    keywords = st.text_area(
        "Target Keywords (one per line)",
        height=200,
        placeholder="Enter keywords here...\nOne per line"
    )

if st.button("Analyze Content", type="primary"):
    if not content or not keywords:
        st.error("Please provide both content and keywords")
    else:
        try:
            with st.spinner("Analyzing content..."):
                # Process keywords
                keyword_list = [k.strip() for k in keywords.split('\n') if k.strip()]
                
                # Get content embedding
                content_embedding = get_embedding(content)
                if content_embedding is None:
                    st.error("Failed to get content embedding")
                    st.stop()
                
                # Calculate similarities
                results = []
                for keyword in keyword_list:
                    keyword_embedding = get_embedding(keyword)
                    if keyword_embedding is not None:
                        similarity = cosine_similarity(content_embedding, keyword_embedding)
                        results.append({
                            "Keyword": keyword,
                            "Similarity Score (%)": round(similarity * 100, 2)
                        })
                
                # Create DataFrame and visualizations
                if results:
                    df = pd.DataFrame(results)
                    df = df.sort_values("Similarity Score (%)", ascending=False)
                    
                    # Create visualization
                    fig = px.bar(
                        df,
                        x="Keyword",
                        y="Similarity Score (%)",
                        title="Semantic Relevance Scores",
                        color="Similarity Score (%)",
                        color_continuous_scale="viridis"
                    )
                    fig.update_layout(
                        xaxis_tickangle=-45,
                        yaxis_range=[0, 100]
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show detailed results
                    st.subheader("Detailed Analysis")
                    st.dataframe(df)
                    
                    # Provide recommendations
                    st.subheader("Content Recommendations")
                    for _, row in df.iterrows():
                        score = row["Similarity Score (%)"]
                        keyword = row["Keyword"]
                        if score < 50:
                            st.warning(f"Low relevance ({score:.1f}%) for '{keyword}' - Consider adding more context about this topic")
                        elif score < 70:
                            st.info(f"Moderate relevance ({score:.1f}%) for '{keyword}' - Could strengthen this topic")
                        else:
                            st.success(f"Strong relevance ({score:.1f}%) for '{keyword}' - Good coverage")
                else:
                    st.error("Failed to analyze keywords")

# Add sidebar with instructions
st.sidebar.markdown("""
### How to use this tool
1. Make sure your embedding server is running locally
2. Paste your content in the left text area
3. Enter your target keywords (one per line)
4. Click 'Analyze Content'
5. Review the semantic relevance scores
6. Check recommendations for improvement

### About Semantic Analysis
This tool uses vector embeddings to analyze how well your content aligns with your target keywords semantically, going beyond simple keyword matching.
""")
