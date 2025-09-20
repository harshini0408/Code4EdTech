import streamlit as st
import re
from sentence_transformers import SentenceTransformer, util
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- LAZY LOADING FOR AI MODELS ---
@st.cache_resource
def get_embedding_model():
    """Loads and caches the Sentence Transformer model."""
    print("Loading embedding model for the first time...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Embedding model loaded.")
    return model

@st.cache_resource
def get_llm():
    """Loads and caches the Gemini LLM."""
    print("Initializing Gemini LLM for the first time...")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2, convert_system_message_to_human=True)
    print("Gemini LLM initialized.")
    return llm

# --- SCORING AND FEEDBACK FUNCTIONS ---

def get_local_semantic_similarity(resume_text, jd_text):
    """Calculates semantic similarity using the lazy-loaded local model."""
    model = get_embedding_model()
    try:
        resume_embedding = model.encode(resume_text, convert_to_tensor=True)
        jd_embedding = model.encode(jd_text, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(resume_embedding, jd_embedding)
        return cosine_scores.item()
    except Exception as e:
        print(f"Error in local similarity: {e}")
        return 0.0

def get_llm_feedback_and_score(resume_text, jd_text):
    """Uses Gemini to generate a score, verdict, and qualitative feedback."""
    llm = get_llm()
    prompt = PromptTemplate.from_template(
        """
        As an expert HR analyst, evaluate the resume against the job description.
        Provide a concise analysis in this exact format, with no extra text:

        Job Description:
        {jd}
        ---
        Resume:
        {resume}
        ---
        Analysis:
        Score: [A single integer from 0 to 100]
        Verdict: [A single word: High, Medium, or Low]
        Missing:
        - [Top 3 critical missing skills or qualifications]
        Suggestions: [A short, actionable paragraph for the student on how to improve.]
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    try:
        response = chain.run(resume=resume_text, jd=jd_text)
        # Parsing logic
        score = int(re.search(r"Score: (\d+)", response).group(1))
        verdict = re.search(r"Verdict: (\w+)", response).group(1)
        missing = re.search(r"Missing:\n(.*?)\nSuggestions:", response, re.DOTALL).group(1).strip()
        suggestions = re.search(r"Suggestions: (.*)", response, re.DOTALL).group(1).strip()
        return {
            "llm_score": score / 100.0,
            "verdict": verdict,
            "missing_elements": [m.strip() for m in missing.split('- ') if m.strip()],
            "suggestions": suggestions
        }
    except Exception:
        return {"llm_score": 0, "verdict": "Error", "missing_elements": [], "suggestions": "AI analysis failed."}

def calculate_advanced_relevance(resume_data, jd_data):
    """Calculates the final hybrid score."""
    # 1. Keyword Score
    resume_skills = set(resume_data.get('skills', []))
    jd_skills = set(jd_data.get('required_skills', []))
    keyword_score = len(resume_skills.intersection(jd_skills)) / len(jd_skills) if jd_skills else 0.5
    
    # 2. Semantic Score
    semantic_score = get_local_semantic_similarity(resume_data['raw_text'], jd_data.get('cleaned_text', ''))
    
    # 3. LLM Score & Feedback
    llm_analysis = get_llm_feedback_and_score(resume_data['raw_text'], jd_data['raw_text'])
    llm_score = llm_analysis.get('llm_score', 0)

    # Weighted Final Score: 20% Keyword, 40% Semantic, 40% LLM
    final_score = min(int(((keyword_score * 0.2) + (semantic_score * 0.4) + (llm_score * 0.4)) * 100), 100)
    
    return {
        'overall_score': final_score,
        'verdict': llm_analysis.get('verdict', 'Low'),
        'missing_elements': llm_analysis.get('missing_elements', []),
        'suggestions': llm_analysis.get('suggestions', '')
    }
