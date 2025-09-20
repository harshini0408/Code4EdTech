import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

def get_resume_embeddings(resume_data_list):
    """Generates embeddings using the cached model."""
    from src.scoring_engine import get_embedding_model
    model = get_embedding_model()
    resume_texts = [resume['raw_text'] for resume in resume_data_list]
    return model.encode(resume_texts) if resume_texts else np.array([])

def perform_clustering(embeddings, num_clusters):
    """Performs K-Means clustering."""
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    return kmeans.fit(embeddings).labels_

def generate_cluster_names(cluster_keywords_map):
    """Uses Gemini to name clusters."""
    from src.scoring_engine import get_llm
    llm = get_llm()
    prompt = PromptTemplate.from_template(
        "Keywords: {keywords}. Generate a 2-3 word professional title for this resume group (e.g., 'Frontend Web Developers')."
    )
    cluster_names = {}
    for cluster_id, keywords in cluster_keywords_map.items():
        try:
            response = llm.invoke(prompt.format(keywords=", ".join(keywords)))
            cluster_names[cluster_id] = response.content.strip().replace('"', '')
        except Exception:
            cluster_names[cluster_id] = f"Cluster {cluster_id + 1}"
    return cluster_names

def analyze_clusters(parsed_resumes):
    """The main clustering pipeline."""
    if not parsed_resumes:
        return parsed_resumes, {}

    embeddings = get_resume_embeddings(parsed_resumes)
    num_resumes = len(parsed_resumes)
    
    if num_resumes < 5:
        cluster_labels = [0] * num_resumes
        num_clusters = 1
    else:
        num_clusters = min(5, max(2, num_resumes // 5))
        cluster_labels = perform_clustering(embeddings, num_clusters)

    all_texts = [res['raw_text'] for res in parsed_resumes]
    cluster_keywords = {}
    for i in range(num_clusters):
        texts_in_cluster = [all_texts[j] for j, label in enumerate(cluster_labels) if label == i]
        if texts_in_cluster:
            words = re.findall(r'\b[a-z]{4,15}\b', " ".join(texts_in_cluster).lower())
            stopwords = {'the', 'and', 'for', 'with', 'our', 'are', 'from'}
            meaningful_words = [word for word in words if word not in stopwords]
            cluster_keywords[i] = [word for word, count in Counter(meaningful_words).most_common(10)]
    
    cluster_names = generate_cluster_names(cluster_keywords)
    
    for i, resume in enumerate(parsed_resumes):
        cluster_id = cluster_labels[i]
        resume['cluster_name'] = cluster_names.get(cluster_id, "General")
        
    return parsed_resumes, cluster_names
