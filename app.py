import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter
import re
from vertexai.language_models import TextEmbeddingModel
import google.cloud.aiplatform as vertex_ai

# Initialize Vertex AI
vertex_ai.init(
    project=st.secrets["GOOGLE_CLOUD_PROJECT"],
    location=st.secrets["GOOGLE_CLOUD_LOCATION"]
)

def get_embedding(text):
    """Get embedding from Google's PaLM API"""
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@latest")
    embeddings = model.get_embeddings([text])
    return embeddings[0].values

def cosine_similarity(vec_a, vec_b):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot_product / (norm_a * norm_b)

def analyze_content_structure(content):
    """Analyze content structure for SEO improvements"""
    # Basic structure analysis
    paragraphs = content.split('\n\n')
    sentences = content.split('.')
    words = content.split()
    
    analysis = {
        'total_words': len(words),
        'avg_paragraph_length': len(words) / len(paragraphs) if paragraphs else 0,
        'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
        'headings': len([p for p in paragraphs if p.strip().startswith('#')])
    }
    return analysis

def get_content_suggestions(content, keyword_scores):
    """Get GPT suggestions for content improvement"""
    top_keywords = sorted(keyword_scores, key=lambda x: x['Similarity Score (%)'], reverse=True)[:3]
    keywords_str = ', '.join([k['Keyword'] for k in top_keywords])
    
    # Create prompt for GPT
    prompt = f"""As an SEO expert, analyze this content for search engine optimization. 
    Target keywords: {keywords_str}
    
    Content: {content[:1000]}...
    
    Provide 3-5 specific, actionable suggestions to improve this content for better search rankings. 
    Focus on:
    1. Semantic relevance to target keywords
    2. Content structure and formatting
    3. Search intent alignment
    4. Missing subtopics or points
    
    Format each suggestion as: 'Suggestion: [brief suggestion] - Reason: [brief explanation]'"""

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating suggestions: {str(e)}"

def analyze_keyword_density(content, keywords):
    """Analyze keyword density and variations"""
    words = content.lower().split()
    total_words = len(words)
    
    density_analysis = {}
    for keyword in keywords:
        keyword_lower = keyword.lower()
        keyword_words = keyword_lower.split()
        
        # Count exact matches
        exact_count = content.lower().count(keyword_lower)
        
        # Count partial matches (for multi-word keywords)
        partial_matches = []
        if len(keyword_words) > 1:
            for word in keyword_words:
                if len(word) > 3:  # Only check significant words
                    word_count = content.lower().count(word)
                    partial_matches.append(f"{word}: {word_count}")
        
        density = (exact_count * len(keyword_words) / total_words * 100) if total_words > 0 else 0
        
        density_analysis[keyword] = {
            'exact_matches': exact_count,
            'density_percentage': round(density, 2),
            'partial_matches': partial_matches if partial_matches else None
        }
    
    return density_analysis

# Set page config
st.set_page_config(page_title="SEO Content Optimizer", layout="wide")

# Add title and description
st.title("🎯 SEO Content Optimizer")
st.markdown("""
This tool helps you optimize your content for search engines by analyzing semantic relevance, 
providing content suggestions, and checking keyword optimization.
""")

# Create two columns for input
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

# Add analyze button
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
                
                # Create tabs for different analyses
                tab1, tab2, tab3 = st.tabs(["Semantic Analysis", "Content Structure", "Optimization Suggestions"])
                
                with tab1:
                    # Display similarity results
                    st.subheader("Keyword Semantic Alignment")
                    
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
                    
                    # Keyword density analysis
                    st.subheader("Keyword Usage Analysis")
                    density_results = analyze_keyword_density(content, keyword_list)
                    for keyword, analysis in density_results.items():
                        st.write(f"**{keyword}**")
                        st.write(f"- Exact matches: {analysis['exact_matches']}")
                        st.write(f"- Density: {analysis['density_percentage']}%")
                        if analysis['partial_matches']:
                            st.write("- Related word usage:")
                            for match in analysis['partial_matches']:
                                st.write(f"  • {match}")
                
                with tab2:
                    # Display content structure analysis
                    st.subheader("Content Structure Analysis")
                    structure = analyze_content_structure(content)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Words", structure['total_words'])
                    with col2:
                        st.metric("Avg. Paragraph Length", f"{structure['avg_paragraph_length']:.1f} words")
                    with col3:
                        st.metric("Avg. Sentence Length", f"{structure['avg_sentence_length']:.1f} words")
                    with col4:
                        st.metric("Headings", structure['headings'])
                
                with tab3:
                    # Display GPT suggestions
                    st.subheader("Content Optimization Suggestions")
                    suggestions = get_content_suggestions(content, results)
                    st.markdown(suggestions)
                
                # Add download button for results
                st.sidebar.subheader("Export Results")
                csv = df.to_csv(index=False)
                st.sidebar.download_button(
                    label="Download Analysis as CSV",
                    data=csv,
                    file_name="seo_analysis_results.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Add usage instructions in sidebar
with st.sidebar:
    st.subheader("How to Use")
    st.markdown("""
    1. Paste your content in the left text area
    2. Enter your target keywords (one per line)
    3. Click 'Analyze Content'
    4. Review the analysis across three tabs:
        - Semantic Analysis
        - Content Structure
        - Optimization Suggestions
    5. Download results as CSV if needed
    """)
    
    st.subheader("About")
    st.markdown("""
    This tool combines AI embeddings and GPT analysis to help optimize your content 
    for search engines. It analyzes semantic relevance, content structure, and 
    provides actionable suggestions for improvement.
    """)
    
    st.subheader("Interpretation Guide")
    st.markdown("""
    - **Similarity Scores**: Higher scores (>70%) indicate strong semantic alignment
    - **Keyword Density**: Aim for natural usage (1-3% typically)
    - **Content Structure**: Balanced metrics indicate well-structured content
    - **Suggestions**: Focus on implementing high-impact recommendations first
    """)
