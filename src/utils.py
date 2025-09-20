import re

def clean_text(text):
    """A simple function to clean text."""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespaces with a single space
    return text.strip()
