from src.utils import clean_text
from src.resume_parser import extract_skills_from_text # Reuse the same skill extractor

def parse_job_description(jd_text):
    """Parses JD text to extract key requirements."""
    cleaned_text = clean_text(jd_text)
    return {
        'raw_text': jd_text,
        'cleaned_text': cleaned_text,
        'required_skills': extract_skills_from_text(cleaned_text) # Extract skills from JD
    }
