# providers/groq_provider.py

import os
import logging
from groq import Groq
from providers.api_provider.py import APIProvider

class GroqProvider(APIProvider):
    def __init__(self, config, requirements):
        super().__init__(config, requirements)
        try:
            self.client = Groq(api_key=config['api_key'])
            self.model_name = config.get('model_name', "llama3-70b-8192")
            self.temperature = config.get('temperature', 0.7)
            self.max_output_tokens = config.get('max_output_tokens', 4096)
            logging.info("GroqProvider initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing GroqProvider: {e}")
            raise

    def process_record(self, record):
        """
        Process the record using the Groq API.
        """
        try:
            prompt = f"""Please process the following data according to these requirements:

{self.requirements}

Here is the data:

{record}

Please provide the processed data in the same format, ensuring that all modifications adhere to the requirements."""

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_output_tokens,
                temperature=self.temperature
            )

            processed_data = response.choices[0].message.content.strip()
            return processed_data
        except Exception as e:
            logging.error(f"Error processing record with GroqProvider: {e}")
            return None
