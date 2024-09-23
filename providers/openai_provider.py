# providers/openai_provider.py

import logging
import openai
from providers.api_provider import APIProvider
from typing import List, Optional, Dict, Any


class OpenAIProvider(APIProvider):
    def __init__(self, config, requirements):
        super().__init__(config, requirements)
        openai.api_key = config['api_key']
        self.model_name = config.get('model_name', "gpt-3.5-turbo")
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_output_tokens', 4096)
        logging.info("OpenAIProvider initialized successfully.")

    def process_record(self, record):
        """
        Process the record using the OpenAI API.
        """
        try:
            prompt = f"""Please process the following data according to these requirements:

            {self.requirements}

            Here is the data:

            {record}

            Please provide the processed data in the same format, ensuring that all modifications adhere to the requirements."""

            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            processed_data = response.choices[0].message.content.strip()
            return processed_data
        except Exception as e:
            logging.error(f"Error processing record with OpenAIProvider: {e}")
            return None
        
    def format_text(self, prompt: str, stop_sequence: Optional[List[str]] = None) -> str:
            """
            Format text using OpenAI's Completion API.

            :param prompt: The prompt to send to OpenAI.
            :param stop_sequence: Optional list of stop sequences to terminate the LLM response.
            :return: The formatted text returned by OpenAI.
            """
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=500,
                    n=1,
                    stop=stop_sequence
                )
                formatted_text = response.choices[0].message.content.strip()
                logging.debug("Received response from OpenAI.")
                return formatted_text
            except Exception as e:
                logging.error(f"OpenAI formatting failed: {e}")
                return ""