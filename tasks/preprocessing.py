# tasks/preprocessing.py

import re
import logging

def remove_pii(text):
    """
    Remove Personally Identifiable Information (PII) from the text.
    Example: Remove email addresses, phone numbers, etc.
    """
    try:
        # Example regex patterns (extend as needed)
        pii_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Emails
            r'\b\d{10}\b',  # Phone numbers
            # Add more patterns as necessary
        ]
        for pattern in pii_patterns:
            text = re.sub(pattern, '[REDACTED]', text)
        return text
    except Exception as e:
        logging.error(f"Error removing PII: {e}")
        return text

def remove_promotional_text(text):
    """
    Remove promotional or irrelevant texts from the content.
    """
    try:
        # Define markers or patterns that signify promotional content
        promo_patterns = [
            r'Nguyên Luật.*',  # Example pattern
            r'Liên hệ*',              # Example pattern
            # Add more patterns as necessary
        ]
        for pattern in promo_patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL)
        return text
    except Exception as e:
        logging.error(f"Error removing promotional text: {e}")
        return text

def generalize_title(title, content):
    """
    Rephrase the title to better generalize the content.
    """
    try:
        # Simple heuristic: Remove specific terms to generalize
        generalized_title = re.sub(r'cấp Giấy chứng nhận ATTP', 'Procedures for Obtaining ATTP Certificate', title)
        return generalized_title
    except Exception as e:
        logging.error(f"Error generalizing title: {e}")
        return title

def preprocess_record(record):
    """
    Apply all preprocessing steps to a single record.
    """
    try:
        # Assuming record is a dictionary
        record['content'] = remove_pii(record['content'])
        record['content'] = remove_promotional_text(record['content'])
        record['title'] = generalize_title(record['title'], record['content'])
        # Add more preprocessing steps as needed
        return record
    except Exception as e:
        logging.error(f"Error preprocessing record: {e}")
        return record
