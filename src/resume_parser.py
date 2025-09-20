import pymupdf
from docx import Document
import re

def extract_text_from_pdf(file):
    """Extracts raw text from a PDF file."""
    try:
        file.seek(0)
        doc = pymupdf.open(stream=file.read(), filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def extract_text_from_docx(file):
    """Extracts raw text from a DOCX file."""
    try:
        file.seek(0)
        doc = Document(file)
        text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
        return text
    except Exception as e:
        print(f"Error reading DOCX: {e}")
        return ""

def parse_resume(text):
    """Parses resume text to extract skills and experience."""
    return {
        "raw_text": text,
        "skills": extract_skills_from_text(text),
        "experience": extract_experience(text)
    }

def extract_skills_from_text(text):
    """Uses regex to find common technical skills."""
    skills_patterns = [
        'python', 'java', 'javascript', 'c\\+\\+', 'c#', 'sql', 'nosql',
        'react', 'angular', 'vue', 'node.js', 'django', 'flask',
        'machine learning', 'deep learning', 'data science', 'nlp',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git',
        'api', 'rest', 'graphql', 'mongodb', 'postgresql', 'mysql'
    ]
    found_skills = set()
    text_lower = text.lower()
    for pattern in skills_patterns:
        if re.search(r'\b' + pattern.replace('.', r'\.') + r'\b', text_lower):
            found_skills.add(pattern)
    return list(found_skills)

def extract_experience(text):
    """Estimates years of experience from text."""
    experience_patterns = [r'(\d+)\s*to\s*(\d+)\s*years', r'(\d+)\+?\s*years?']
    text_lower = text.lower()
    all_found_years = []
    for pattern in experience_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if isinstance(match, tuple):
                all_found_years.extend([int(y) for y in match])
            else:
                all_found_years.append(int(match))
    return max(all_found_years) if all_found_years else 0
