import streamlit as st
import numpy as np
from openai import OpenAI
import pandas as pd
import plotly.express as px

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def get_embedding(text):
    """Get embedding from OpenAI API"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(vec_a, vec_b):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot_product / (norm_a * norm_b)

# Set page config
st.set_page_config(page_title="Keyword Similarity Analyzer", layout="wide")

# Add title and description
st.title("📊 Keyword Similarity Analyzer")
st.markdown("""
This tool helps you optimize your content for SEO by comparing it against potential target keywords.
It uses OpenAI's embeddings to calculate semantic similarity between your content and keywords.
""")

# Create two columns for input
col1, col2 = st.columns([2, 1])

with col1:
    # Content input
    content = st.text_area(
        "Your Content",
        height=200,
        placeholder="Paste your content here..."
    )

with col2:
    # Keyword input
    keywords = st.text_area(
        "Target Keywords (one per line)",
        height=200,
        placeholder="Enter keywords here...\nOne per line"
    )

# Add analyze button
if st.button("Analyze Similarity", type="primary"):
    if not content or not keywords:
        st.error("Please provide both content and keywords")
    else:
        try:
            with st.spinner("Analyzing content..."):
                # Process keywords
                keyword_list = [k.strip() for k in keywords.split('\n') if k.strip()]
                
                # Get content embedding
                content_embedding = get_embedding(content)
                
                # Calculate similarities
                results = []
                for keyword in keyword_list:
                    keyword_embedding = get_embedding(keyword)
                    similarity = cosine_similarity(content_embedding, keyword_embedding)
                    results.append({
                        "Keyword": keyword,
                        "Similarity Score (%)": round(similarity * 100, 2)
                    })
                
                # Create DataFrame
                df = pd.DataFrame(results)
                df = df.sort_values("Similarity Score (%)", ascending=False)
                
                # Display results
                st.subheader("Results")
                
                # Create two columns for visualization and table
                viz_col, table_col = st.columns([2, 1])
                
                with viz_col:
                    # Create bar chart
                    fig = px.bar(
                        df,
                        x="Keyword",
                        y="Similarity Score (%)",
                        title="Keyword Similarity Scores",
                        color="Similarity Score (%)",
                        color_continuous_scale="viridis"
                    )
                    fig.update_layout(
                        xaxis_tickangle=-45,
                        yaxis_range=[0, 100]
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with table_col:
                    # Display table
                    st.dataframe(
                        df.style.format({"Similarity Score (%)": "{:.2f}"}),
                        hide_index=True,
                        use_container_width=True
                    )
                
                # Add download button for results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="keyword_similarity_results.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Add usage instructions in sidebar
with st.sidebar:
    st.subheader("How to Use")
    st.markdown("""
    1. Paste your content in the left text area
    2. Enter your target keywords in the right text area (one per line)
    3. Click 'Analyze Similarity'
    4. View results in the interactive chart and table
    5. Download results as CSV if needed
    """)
    
    st.subheader("About")
    st.markdown("""
    This tool uses OpenAI's text embeddings to calculate semantic similarity 
    between your content and potential keywords. Higher similarity scores 
    indicate better content-keyword alignment.
    """)
    
    st.subheader("Tips")
    st.markdown("""
    - Try different variations of your keywords
    - Compare competing keywords to find the best match
    - Consider scores above 80% as strong matches
    - Look for patterns in high-scoring keywords
    """)
