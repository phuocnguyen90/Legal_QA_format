# providers/groq_provider.py

import logging
import json
import gzip
from groq import Groq
from providers.api_provider import APIProvider
from typing import List, Dict, Any, Optional
import re

    
class GroqProvider(APIProvider):
    def __init__(self, config: Dict[str, Any], requirements: str):
        super().__init__(config, requirements)
        try:
            api_key = config['api_key']
            if not api_key:
                logging.error("Groq API key is missing.")
                raise ValueError("Groq API key is missing.")
            self.client = Groq(api_key=api_key)  
            self.model_name = config.get('model_name', "llama3-70b-8192")
            self.temperature = config.get('temperature', 0.7)
            self.max_output_tokens = config.get('max_output_tokens', 4096)
            logging.info("GroqProvider initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize GroqProvider: {e}")
            raise
    
    def process_record(self, record: Dict[str, Any]) -> Optional[str]:
        """
        Process the record using the Groq API.
        
        :param record: Dictionary representing the record.
        :return: Processed data as a JSON string or None if processing fails.
        """
        try:
            # Step 1: Prepare the prompt
            prompt = f"""Please process the following data according to these requirements:

            {self.requirements}

            Here is the data:

            {json.dumps(record, ensure_ascii=False, indent=2)}

            Please provide the processed data in JSON format, ensuring that all modifications adhere to the requirements."""

            logging.debug("Sending request to Groq API.")

            # Step 2: Send the request to Groq API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_output_tokens
            )
            logging.debug("Received response from Groq API.")

            # Step 3: Validate the response
            if not response or not hasattr(response, 'choices') or not response.choices:
                logging.error("Invalid or empty response structure from Groq API.")
                return None

            # Step 4: Extract content
            processed_data = response.choices[0].message.content.strip()
            if not processed_data:
                logging.error("Empty content received in the response from Groq API.")
                return None

            logging.debug("Processing response content.")
            # logging.debug(f"Full API response: {response}")

            # Step 5: Extract JSON from the response content
            # The content is wrapped in triple backticks and needs to be extracted
            json_match = re.search(r'```(.*?)```', processed_data, re.DOTALL)
            if json_match:
                json_content = json_match.group(1).strip()
                try:
                    # Ensure the extracted content is a valid JSON
                    processed_json = json.loads(json_content)
                    return json.dumps(processed_json, ensure_ascii=False)
                except json.JSONDecodeError as e:
                    logging.error(f"JSON decoding error in extracted content: {e}")
                    return None
            else:
                logging.error("No JSON content found in the API response.")
                return None

        except Exception as e:
            logging.error(f"Error processing record with GroqProvider: {e}")
            return None
        
    def format_text(self, prompt: str, stop_sequence: Optional[List[str]] = None) -> str:
        """
        Format text using Groq's API.

        :param prompt: The prompt to send to Groq.
        :param stop_sequence: Optional list of stop sequences to terminate the LLM response.
        :return: The formatted text returned by Groq.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_output_tokens
            )
            logging.debug("Received response from Groq.")
            
            # Extract content
            processed_data = response.choices[0].message.content.strip()
            if not processed_data:
                logging.error("Empty content received in the response from Groq API.")
                return ""
            
            # Extract JSON from the response content if formatted as JSON
            json_match = re.search(r'```json\s*(.*?)\s*```', processed_data, re.DOTALL)
            if json_match:
                json_content = json_match.group(1).strip()
                try:
                    # Ensure the extracted content is valid JSON
                    processed_json = json.loads(json_content)
                    return json.dumps(processed_json, ensure_ascii=False)
                except json.JSONDecodeError as e:
                    logging.error(f"JSON decoding error in extracted content: {e}")
                    return ""
            else:
                # If not JSON, return as is
                return processed_data
        except Exception as e:
            logging.error(f"Groq formatting failed: {e}")
            return ""