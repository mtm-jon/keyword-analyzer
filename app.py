import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from vertexai.language_models import TextEmbeddingModel
import google.cloud.aiplatform as vertex_ai
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import re
import re

# Initialize Vertex AI
vertex_ai.init(
    project=st.secrets["GOOGLE_CLOUD_PROJECT"],
    location=st.secrets["GOOGLE_CLOUD_LOCATION"]
)

def get_embedding(text):
    """Get embedding from local Vertex AI server"""
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

def chunk_content(text, chunk_size=3):
    """Split content into overlapping chunks"""
    sentences = simple_sentence_tokenize(text)
    chunks = []
    for i in range(0, len(sentences) - chunk_size + 1):
        chunk = ' '.join(sentences[i:i + chunk_size])
        chunks.append({
            'text': chunk,
            'position': i,
            'sentences': sentences[i:i + chunk_size]
        })
    return chunks

def detect_content_anomalies(chunk_embeddings, threshold=2):
    """Detect unusual content sections using vector analysis"""
    vectors = np.array([c['embedding'] for c in chunk_embeddings])
    
    # Calculate pairwise distances
    distances = np.mean(np.abs(vectors - np.mean(vectors, axis=0)), axis=1)
    
    # Find anomalies (chunks that are unusually different)
    anomalies = []
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    for i, distance in enumerate(distances):
        if distance > mean_dist + threshold * std_dist:
            anomalies.append({
                'chunk': chunk_embeddings[i]['text'],
                'deviation_score': (distance - mean_dist) / std_dist
            })
    
    return anomalies

def analyze_content_diversity(chunk_embeddings, n_clusters=3):
    """Analyze content diversity using clustering"""
    vectors = np.array([c['embedding'] for c in chunk_embeddings])
    
    # Normalize vectors
    scaler = StandardScaler()
    vectors_normalized = scaler.fit_transform(vectors)
    
    # Cluster content
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(vectors_normalized)
    
    # Analyze clusters
    cluster_analysis = []
    for i in range(n_clusters):
        cluster_chunks = [chunk_embeddings[j]['text'] for j in range(len(clusters)) if clusters[j] == i]
        cluster_analysis.append({
            'cluster_id': i,
            'size': len(cluster_chunks),
            'chunks': cluster_chunks,
            'coverage': len(cluster_chunks) / len(chunk_embeddings)
        })
    
    return cluster_analysis

def simple_sentence_tokenize(text):
    """Simple but robust sentence tokenization"""
    # Handle multiple punctuation marks
    sentence_endings = r'[.!?]+'
    # Split on sentence endings followed by space or newline
    sentences = re.split(f'{sentence_endings}[\s\n]+', text)
    # Clean up and filter empty sentences
    return [s.strip() + '.' for s in sentences if s.strip()]

def analyze_serp_features(content):
    """Analyze content for potential SERP feature opportunities"""
    features = {
        'featured_snippet': {
            'detected': False,
            'type': None,
            'content': None,
            'confidence': 0
        },
        'list_potential': {
            'detected': False,
            'type': None,
            'content': None,
            'confidence': 0
        },
        'table_potential': {
            'detected': False,
            'content': None,
            'confidence': 0
        },
        'qa_potential': {
            'detected': False,
            'questions': [],
            'confidence': 0
        },
        'how_to_potential': {
            'detected': False,
            'steps': [],
            'confidence': 0
        }
    }
    
    # Check for list potential
    list_markers = ['1.', '•', '-', '*', '1)', 'Step 1', 'First']
    list_lines = [line for line in content.split('\n') if any(line.strip().startswith(marker) for marker in list_markers)]
    if list_lines:
        features['list_potential'] = {
            'detected': True,
            'type': 'ordered' if any(line.strip()[0].isdigit() for line in list_lines) else 'unordered',
            'content': list_lines,
            'confidence': min(len(list_lines) / 3, 1.0)  # Higher confidence with more list items
        }
    
    # Check for table potential
    if '|' in content or '\t' in content:
        features['table_potential'] = {
            'detected': True,
            'content': 'Table structure detected',
            'confidence': 0.8
        }
    
    # Check for Q&A potential
    question_markers = ['?', 'What', 'How', 'Why', 'When', 'Where', 'Who']
    questions = [s for s in content.split('\n') if any(s.strip().startswith(q) for q in question_markers) and '?' in s]
    if questions:
        features['qa_potential'] = {
            'detected': True,
            'questions': questions,
            'confidence': min(len(questions) / 2, 1.0)
        }
    
    # Check for How-to potential
    step_markers = ['Step', 'First', 'Second', 'Third', 'Finally', 'Next']
    steps = [s for s in content.split('\n') if any(s.strip().startswith(m) for m in step_markers)]
    if steps:
        features['how_to_potential'] = {
            'detected': True,
            'steps': steps,
            'confidence': min(len(steps) / 3, 1.0)
        }
    
    # Check for featured snippet potential
    # Prioritize definitions, short explanations, and clear answers
    sentences = [s for s in simple_sentence_tokenize(content) if len(s.strip()) > 20 and len(s.strip()) < 300]
    definition_starters = ['is a', 'refers to', 'means', 'defines', 'consists of']
    for sentence in sentences:
        if any(marker in sentence.lower() for marker in definition_starters):
            features['featured_snippet'] = {
                'detected': True,
                'type': 'definition',
                'content': sentence,
                'confidence': 0.9
            }
            break
    
    return features

def compare_serp_features(your_features, competitor_features):
    """Compare SERP feature opportunities between your content and competitor content"""
    comparison = {
        'missing_features': [],
        'matching_features': [],
        'your_advantages': [],
        'recommendations': []
    }
    
    # Compare each feature type
    for feature in your_features.keys():
        if competitor_features[feature]['detected'] and not your_features[feature]['detected']:
            comparison['missing_features'].append({
                'feature': feature,
                'competitor_confidence': competitor_features[feature]['confidence']
            })
            comparison['recommendations'].append(f"Add {feature.replace('_', ' ')} structure to match competitor")
        elif your_features[feature]['detected'] and not competitor_features[feature]['detected']:
            comparison['your_advantages'].append({
                'feature': feature,
                'your_confidence': your_features[feature]['confidence']
            })
        elif your_features[feature]['detected'] and competitor_features[feature]['detected']:
            comparison['matching_features'].append({
                'feature': feature,
                'your_confidence': your_features[feature]['confidence'],
                'competitor_confidence': competitor_features[feature]['confidence']
            })
    
    return comparison
    """Analyze competitor content in vector space"""
    # Convert embeddings to numpy arrays for vector operations
    your_vectors = np.array([e['embedding'] for e in your_embeddings])
    competitor_vectors = np.array([e['embedding'] for e in competitor_embeddings])
    keyword_vectors = np.array([e['embedding'] for e in keyword_embeddings])
    
    # Calculate coverage scores
    your_coverage = np.mean([
        np.max([
            np.dot(kv, yv) / (np.linalg.norm(kv) * np.linalg.norm(yv))
            for yv in your_vectors
        ])
        for kv in keyword_vectors
    ])
    
    competitor_coverage = np.mean([
        np.max([
            np.dot(kv, cv) / (np.linalg.norm(kv) * np.linalg.norm(cv))
            for cv in competitor_vectors
        ])
        for kv in keyword_vectors
    ])
    
    # Find unique angles (topics) covered by competitor but not you
    gaps = []
    for i, comp_vec in enumerate(competitor_vectors):
        max_similarity_competitor = max([
            np.dot(comp_vec, yv) / (np.linalg.norm(comp_vec) * np.linalg.norm(yv))
            for yv in your_vectors
        ])
        if max_similarity_competitor < 0.7:  # Threshold for considering it a gap
            gaps.append({
                'text': competitor_embeddings[i]['text'],
                'uniqueness_score': 1 - max_similarity_competitor
            })
    
    # Find your unique strengths
    strengths = []
    for i, your_vec in enumerate(your_vectors):
        max_similarity_yours = max([
            np.dot(your_vec, cv) / (np.linalg.norm(your_vec) * np.linalg.norm(cv))
            for cv in competitor_vectors
        ])
        if max_similarity_yours < 0.7:  # Threshold for considering it unique
            strengths.append({
                'text': your_embeddings[i]['text'],
                'uniqueness_score': 1 - max_similarity_yours
            })
    
    return {
        'your_coverage': your_coverage,
        'competitor_coverage': competitor_coverage,
        'gaps': sorted(gaps, key=lambda x: x['uniqueness_score'], reverse=True),
        'strengths': sorted(strengths, key=lambda x: x['uniqueness_score'], reverse=True)
    }

def recommend_similar_topics(keyword_embeddings, competitor_analysis=None, num_recommendations=5):
    """Generate topic recommendations based on vector similarity and competitor analysis"""
    # Pre-defined topic vectors (could be expanded)
    common_topics = {
        'product_reviews': get_embedding("detailed product reviews and comparisons"),
        'how_to_guides': get_embedding("step by step tutorials and guides"),
        'industry_news': get_embedding("latest industry news and updates"),
        'case_studies': get_embedding("detailed case studies and success stories"),
        'expert_interviews': get_embedding("interviews with industry experts"),
        'troubleshooting': get_embedding("common problems and solutions"),
        'best_practices': get_embedding("industry best practices and tips"),
        'research_findings': get_embedding("research results and analysis")
    }
    
    recommendations = []
    for keyword in keyword_embeddings:
        topic_similarities = {}
        for topic, topic_vector in common_topics.items():
            similarity = np.dot(keyword['embedding'], topic_vector) / (
                np.linalg.norm(keyword['embedding']) * np.linalg.norm(topic_vector)
            )
            topic_similarities[topic] = similarity
        
        # Get top recommendations
        top_topics = sorted(topic_similarities.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]
        recommendations.append({
            'keyword': keyword['keyword'],
            'recommended_topics': [{'topic': t[0], 'relevance': t[1]} for t in top_topics]
        })
    
    return recommendations

# Streamlit UI
st.set_page_config(page_title="Advanced SEO Vector Analysis", layout="wide")

st.title("🎯 Advanced Vector-Based SEO Analyzer")
st.markdown("""
This tool uses advanced vector operations to analyze your content from multiple angles:
- Content-Keyword Alignment
- Topic Diversity Analysis
- Anomaly Detection
- Content Recommendations
""")

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
        height=100,
        placeholder="Enter keywords here...\nOne per line"
    )
    competitor_content = st.text_area(
        "Competitor Content (optional)",
        height=100,
        placeholder="Paste competitor content here..."
    )

if st.button("Run Advanced Analysis", type="primary"):
    if not content or not keywords:
        st.error("Please provide both content and keywords")
    else:
        try:
            with st.spinner("Performing advanced vector analysis..."):
                # Process content and keywords
                keyword_list = [k.strip() for k in keywords.split('\n') if k.strip()]
                content_chunks = chunk_content(content)
                
                # Generate embeddings
                content_embeddings = [{
                    'text': chunk['text'],
                    'position': chunk['position'],
                    'embedding': get_embedding(chunk['text'])
                } for chunk in content_chunks]
                
                keyword_embeddings = [{
                    'keyword': k,
                    'embedding': get_embedding(k)
                } for k in keyword_list]
                
                # Analyze SERP features
                your_serp_features = analyze_serp_features(content)
                competitor_serp_features = analyze_serp_features(competitor_content) if competitor_content else None
                
                if competitor_content:
                    serp_comparison = compare_serp_features(your_serp_features, competitor_serp_features)
                
                # Process competitor content if provided
                competitor_embeddings = []
                competitor_analysis = None
                if competitor_content:
                    competitor_chunks = chunk_content(competitor_content)
                    competitor_embeddings = [{
                        'text': chunk['text'],
                        'position': chunk['position'],
                        'embedding': get_embedding(chunk['text'])
                    } for chunk in competitor_chunks]
                    competitor_analysis = analyze_competitor_content(
                        content_embeddings,
                        competitor_embeddings,
                        keyword_embeddings
                    )
                
                # Run analyses
                anomalies = detect_content_anomalies(content_embeddings)
                diversity_analysis = analyze_content_diversity(content_embeddings)
                topic_recommendations = recommend_similar_topics(
                    keyword_embeddings,
                    competitor_analysis
                )
                
                # Display results in tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "Content Structure",
                    "Anomaly Detection",
                    "Topic Diversity",
                    "Content Recommendations",
                    "Competitor Analysis"
                ])
                
                with tab1:
                    st.subheader("Content Structure Analysis")
                    
                    # Visualize chunk relationships
                    vectors = np.array([c['embedding'] for c in content_embeddings])
                    similarities = np.corrcoef(vectors)
                    
                    fig = px.imshow(
                        similarities,
                        title="Content Chunk Similarity Matrix",
                        color_continuous_scale="viridis"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    st.subheader("Content Anomalies")
                    if anomalies:
                        st.warning("Found potential content inconsistencies:")
                        for anomaly in anomalies:
                            with st.expander(f"Deviation Score: {anomaly['deviation_score']:.2f}"):
                                st.write(anomaly['chunk'])
                                st.write("This section's style or topic differs significantly from the rest.")
                    else:
                        st.success("No significant content anomalies detected!")
                
                with tab3:
                    st.subheader("Topic Diversity Analysis")
                    
                    # Visualize cluster distribution
                    cluster_data = pd.DataFrame([{
                        'Cluster': f"Topic Cluster {c['cluster_id'] + 1}",
                        'Coverage': c['coverage'] * 100
                    } for c in diversity_analysis])
                    
                    fig = px.pie(
                        cluster_data,
                        values='Coverage',
                        names='Cluster',
                        title="Content Topic Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    for cluster in diversity_analysis:
                        with st.expander(f"Topic Cluster {cluster['cluster_id'] + 1} ({cluster['coverage']:.1%} of content)"):
                            st.write("Representative sections:")
                            for chunk in cluster['chunks'][:3]:
                                st.write(f"- {chunk}")
                
                with tab4:
                    st.subheader("Content Recommendations")
                    for rec in topic_recommendations:
                        st.write(f"For keyword: **{rec['keyword']}**")
                        st.write("Recommended related topics to cover:")
                        for topic in rec['recommended_topics']:
                            st.write(f"- {topic['topic'].replace('_', ' ').title()}: {topic['relevance']:.1%} relevance")
                        st.write("---")
                
                with tab5:
                    st.subheader("Competitor Analysis")
                    if competitor_analysis:
                        # SERP Features Analysis
                        st.subheader("🎯 SERP Feature Analysis")
                        
                        # Create tabs for different SERP analyses
                        serp_tab1, serp_tab2, serp_tab3 = st.tabs([
                            "Feature Comparison",
                            "Missing Opportunities",
                            "Your Advantages"
                        ])
                        
                        with serp_tab1:
                            st.markdown("### Feature Comparison")
                            
                            # Create comparison data
                            feature_data = []
                            for feature in your_serp_features:
                                feature_data.append({
                                    'Feature': feature.replace('_', ' ').title(),
                                    'Your Content': your_serp_features[feature]['confidence'] * 100 if your_serp_features[feature]['detected'] else 0,
                                    'Competitor': competitor_serp_features[feature]['confidence'] * 100 if competitor_serp_features[feature]['detected'] else 0
                                })
                            
                            df_features = pd.DataFrame(feature_data)
                            
                            # Create bar chart
                            fig = px.bar(
                                df_features,
                                x='Feature',
                                y=['Your Content', 'Competitor'],
                                title="SERP Feature Potential",
                                barmode='group'
                            )
                            fig.update_layout(yaxis_range=[0, 100])
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with serp_tab2:
                            st.markdown("### Missing SERP Opportunities")
                            if serp_comparison['missing_features']:
                                for feature in serp_comparison['missing_features']:
                                    with st.expander(f"📌 {feature['feature'].replace('_', ' ').title()}"):
                                        st.write(f"Competitor confidence: {feature['competitor_confidence']:.1%}")
                                        if feature['feature'] == 'featured_snippet':
                                            st.write("Recommendation: Add clear definitions or concise explanations")
                                        elif feature['feature'] == 'list_potential':
                                            st.write("Recommendation: Structure content with clear bullet points or numbered lists")
                                        elif feature['feature'] == 'qa_potential':
                                            st.write("Recommendation: Include relevant questions and answers")
                                        elif feature['feature'] == 'how_to_potential':
                                            st.write("Recommendation: Add step-by-step instructions")
                            else:
                                st.success("You're not missing any major SERP features!")
                        
                        with serp_tab3:
                            st.markdown("### Your SERP Advantages")
                            if serp_comparison['your_advantages']:
                                for feature in serp_comparison['your_advantages']:
                                    with st.expander(f"✨ {feature['feature'].replace('_', ' ').title()}"):
                                        st.write(f"Your confidence: {feature['your_confidence']:.1%}")
                                        st.write("This is a unique SERP opportunity in your content!")
                            else:
                                st.info("Focus on adding unique SERP features to stand out!")
                        
                        # Coverage comparison
                        coverage_data = pd.DataFrame([
                            {'Content': 'Your Content', 'Coverage': competitor_analysis['your_coverage'] * 100},
                            {'Content': 'Competitor Content', 'Coverage': competitor_analysis['competitor_coverage'] * 100}
                        ])
                        
                        fig = px.bar(
                            coverage_data,
                            x='Content',
                            y='Coverage',
                            title="Keyword Coverage Comparison",
                            color='Content',
                            color_discrete_sequence=["#4CAF50", "#2196F3"]
                        )
                        fig.update_layout(yaxis_range=[0, 100])
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Content gaps
                        if competitor_analysis['gaps']:
                            st.subheader("Content Gaps (Competitor Advantages)")
                            st.markdown("""
                            These are topics or aspects that your competitor covers but your content doesn't address fully.
                            The uniqueness score indicates how different this content is from yours.
                            """)
                            for gap in competitor_analysis['gaps']:
                                with st.expander(f"Gap (Uniqueness: {gap['uniqueness_score']:.2%})"):
                                    st.write(gap['text'])
                                    st.write("Consider addressing this topic in your content.")
                        
                        # Your unique strengths
                        if competitor_analysis['strengths']:
                            st.subheader("Your Unique Strengths")
                            st.markdown("""
                            These are topics or aspects where your content provides unique value 
                            compared to your competitor.
                            """)
                            for strength in competitor_analysis['strengths']:
                                with st.expander(f"Strength (Uniqueness: {strength['uniqueness_score']:.2%})"):
                                    st.write(strength['text'])
                                    st.write("This is a unique angle in your content.")
                    else:
                        st.info("Add competitor content to see comparative analysis.")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

with st.sidebar:
    st.subheader("About Vector Operations")
    st.markdown("""
    This tool uses four advanced vector operations:

    1. **Anomaly Detection**
    - Identifies content sections that differ significantly
    - Helps spot inconsistencies in style or topic
    
    2. **Topic Diversity**
    - Clusters content into topic groups
    - Shows distribution of subjects
    
    3. **Content Structure**
    - Analyzes relationships between content sections
    - Visualizes content flow and cohesion
    
    4. **Topic Recommendations**
    - Suggests related topics based on vector similarity
    - Helps expand content coverage
    """)
    
    st.subheader("Interpretation Guide")
    st.markdown("""
    - **Anomaly Scores** > 2.0 indicate significant deviation
    - **Topic Clusters** should be relatively balanced
    - **Similarity Matrix** shows content flow (darker = more similar)
    - **Topic Relevance** > 80% indicates strong alignment
    """)
