# providers/gemini_provider.py

import logging
import time
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content
from providers.api_provider import APIProvider


class GeminiProvider(APIProvider):
    def __init__(self, config, requirements):
        super().__init__(config, requirements)
        self.api_key = config['api_key']
        self.model_name = config.get('model_name', "gemini-1.5-flash")
        self.temperature = config.get('temperature', 1)
        self.top_p = config.get('top_p', 0.95)
        self.top_k = config.get('top_k', 64)
        self.max_output_tokens = config.get('max_output_tokens', 8192)
        self.delay_between_requests = config.get('delay_between_requests', 1)

        genai.configure(api_key=self.api_key)

        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "max_output_tokens": self.max_output_tokens,
                "response_schema": content.Schema(
                    type=content.Type.OBJECT,
                    properties={
                        "id": content.Schema(type=content.Type.NUMBER),
                        "title": content.Schema(type=content.Type.STRING),
                        "published_date": content.Schema(type=content.Type.STRING),
                        "categories": content.Schema(
                            type=content.Type.ARRAY,
                            items=content.Schema(type=content.Type.STRING),
                        ),
                        "content": content.Schema(type=content.Type.STRING),
                    },
                ),
                "response_mime_type": "application/json",
            },
            system_instruction=self.requirements
        )

        self.chat_session = self.model.start_chat()
        logging.info("GeminiProvider initialized successfully.")

    def process_record(self, record):
        """
        Process the record using the Google Gemini API.
        """
        try:
            response = self.chat_session.send_message(record)
            processed_data = response.text.strip()
            time.sleep(self.delay_between_requests)  # Respect rate limits
            return processed_data
        except Exception as e:
            logging.error(f"Error processing record with GeminiProvider: {e}")
            return None
