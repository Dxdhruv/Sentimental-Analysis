import re

def basic_text_cleaning(text):
    text = re.sub(r'\s+', ' ', text.lower()).strip()
    return text