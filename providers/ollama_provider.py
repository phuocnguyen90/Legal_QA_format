# providers/ollama_provider.py

import logging
import subprocess
import json
from providers.api_provider import APIProvider


class OllamaProvider(APIProvider):
    def __init__(self, config, requirements):
        super().__init__(config, requirements)
        self.model_path = config['model_path']
        self.temperature = config.get('temperature', 0.7)
        self.max_output_tokens = config.get('max_output_tokens', 4096)
        logging.info("OllamaProvider initialized successfully.")

    def process_record(self, record):
        """
        Process the record using the local Ollama LLM.
        """
        try:
            prompt = f"""Please process the following data according to these requirements:

        {self.requirements}

        Here is the data:

        {record}

        Please provide the processed data in the same format, ensuring that all modifications adhere to the requirements."""

            # Example command to interact with Ollama (adjust as needed)
            process = subprocess.Popen(
                ['ollama', 'prompt', self.model_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate(input=prompt)

            if process.returncode != 0:
                logging.error(f"OllamaProvider error: {stderr}")
                return None

            processed_data = stdout.strip()
            return processed_data
        except Exception as e:
            logging.error(f"Error processing record with OllamaProvider: {e}")
            return None
