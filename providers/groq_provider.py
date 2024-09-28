# providers/groq_provider.py

import logging
from typing import Optional, List, Dict, Any
from groq import Groq  # Ensure the Groq SDK is installed
from providers.api_provider import APIProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GroqProvider(APIProvider):
    """
    A modular provider for interacting with the Groq LLM API.
    """

    def __init__(self, config: Dict[str, Any], requirements: str):
        """
        Initialize the GroqProvider with the specified configuration.

        :param config: Configuration dictionary containing API keys and settings.
        :param requirements: Preprocessing requirements as a string.
        """
        super().__init__(config, requirements)
        try:
            api_key = config.get('api_key')
            if not api_key:
                logger.error("Groq API key is missing.")
                raise ValueError("Groq API key is missing.")
            self.client = Groq(api_key=api_key)
            self.model_name = config.get('model_name', "llama3-70b-8192")
            self.temperature = config.get('temperature', 0.7)
            self.max_output_tokens = config.get('max_output_tokens', 4096)
            logger.info("GroqProvider initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize GroqProvider: {e}")
            raise

    def send_message(self, prompt: str, stop_sequence: Optional[List[str]] = None) -> Optional[str]:
        """
        Send a message to the Groq API and retrieve the response.

        :param prompt: The prompt to send to Groq.
        :param stop_sequence: Optional list of stop sequences to terminate the LLM response.
        :return: The response content from Groq or None if the call fails.
        """
        try:
            logger.debug("Sending prompt to Groq API.")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                stop=stop_sequence
            )
            logger.debug("Received response from Groq API.")

            if not response or not hasattr(response, 'choices') or not response.choices:
                logger.error("Invalid or empty response structure from Groq API.")
                return None

            content = response.choices[0].message.content.strip()
            if not content:
                logger.error("Empty content received in the response from Groq API.")
                return None

            return content

        except Exception as e:
            logger.error(f"Error during Groq API call: {e}")
            return None
