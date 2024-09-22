# utils/llm_client.py

import logging
import openai
# logging.getLogger(__name__)

class LLMClient:
    def __init__(self, api_key: str, model: str = "text-davinci-003"):
        self.api_key = api_key
        self.model = model
        openai.api_key = self.api_key
    
    def format_text(self, raw_text: str) -> str:
        """
        Format unformatted text into structured tagged text using the LLM.
        
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
                engine=self.model,
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
