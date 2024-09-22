# utils/llm_processor.py

import logging
import openai
import os
# logging.getLogger(__name__)

# Ensure that the OPENAI_API_KEY is set in environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

def format_unformatted_text(raw_text: str) -> str:
    """
    Use an LLM to format unformatted text into a structured tagged format.
    
    :param raw_text: The raw unformatted text.
    :return: Formatted tagged text.
    """
    try:
        prompt = f"""You are a data formatter. Convert the following unformatted text into a structured format with tags as shown below:

Example:
<id=1>
<title>Sample Title</title>
<published_date>2024-09-22</published_date>
<categories><Category1><Category2></categories>
<content>
Sample content here.
</content>
</id=1>

Unformatted Text:
{raw_text}

Formatted Text:"""
        
        response = openai.Completion.create(
            engine="gpt-3.5-turbo",  # Or any other suitable model
            prompt=prompt,
            max_tokens=500,
            temperature=0.3,
            n=1,
            stop=["Formatted Text:"]
        )
        
        formatted_text = response.choices[0].text.strip()
        return formatted_text

    except Exception as e:
        logging.error(f"LLM formatting failed: {e}")
        return ""
