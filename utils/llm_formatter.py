# utils/llm_client.py

import logging
import json
from typing import Optional, Dict, Any
from providers import ProviderFactory  
from providers.openai_provider import OpenAIProvider
from providers.groq_provider import GroqProvider
from providers.api_provider import APIProvider
# logging.getLogger(__name__)


class LLMFormatter:
    """
    Unified LLM Formatter supporting multiple formatting modes and providers.
    """

    def __init__(self, config: Dict[str, Any], provider: str = "openai"):
        """
        Initialize the LLMFormatter with the specified provider.

        :param config: Configuration dictionary containing API keys and settings.
        :param provider: The LLM provider to use ("openai", "groq", etc.).
        """
        self.config = config
        self.provider_name = provider.lower()
        self.provider = self._initialize_provider()
        logging.info(f"LLMFormatter initialized with provider '{self.provider_name}'.")

    
    def _initialize_provider(self):
        """
        Initialize the API provider based on the configuration.

        :return: An instance of the API provider.
        """
        requirements = self.config.get('processing', {}).get('pre_process_requirements', "")
        provider_name = self.config.get('provider', '').strip()
        if not provider_name:
            logging.error("API provider not specified in the configuration.")
            raise ValueError("API provider not specified in the configuration.")

        provider_config = self.config.get(provider_name, {})
        if not provider_config:
            logging.error(f"No configuration found for provider '{provider_name}'.")
            raise ValueError(f"No configuration found for provider '{provider_name}'.")

        requirements = self.config.get('tasks', {}).get('pre_process_requirements', "")
        if not requirements:
            logging.error("No 'pre_process_requirements' found under 'tasks' in configuration.")
            raise ValueError("No 'pre_process_requirements' found under 'tasks' in configuration.")

        try:
            provider = ProviderFactory.get_provider(provider_name, provider_config, requirements)
            logging.info(f"Initialized provider: {provider_name}")
            return provider
        except Exception as e:
            logging.error(f"Failed to initialize provider '{provider_name}': {e}")
            raise

    

    def format_text(self, raw_text: str, mode: str = "tagged", provider: Optional[str] = None) -> str:
        """
        Format raw text into structured formats using the specified mode and provider.

        :param raw_text: The raw unformatted text.
        :param mode: The formatting mode ("tagged" or "json"). Defaults to "tagged".
        :param provider: Optional provider override ("openai", "groq", etc.).
        :return: Formatted text as per the specified mode.
        """
        current_provider = self.provider
        if provider:
            # Initialize the specified provider
            temp_formatter = LLMFormatter(config=self.config, provider=provider)
            current_provider = temp_formatter.provider
            if not current_provider:
                logging.error(f"Failed to initialize provider '{provider}'. Using default provider '{self.provider_name}'.")
                current_provider = self.provider

        if not current_provider:
            logging.error("No valid LLM provider available for formatting.")
            return ""

        # Define prompts based on the formatting mode
        if mode == "tagged":
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
            stop_sequence = ["Formatted Text:"]
        
        elif mode == "json":
            prompt = f"""You are a data formatter. Convert the following unformatted text into a structured JSON format adhering to the provided schema.

            Schema:
            {json.dumps(self.config.get('processing', {}).get('json_schema', {}), indent=2)}

            Unformatted Text:
            {raw_text}

            Formatted JSON:"""
            stop_sequence = ["Formatted JSON:"]
        
        else:
            logging.error(f"Unsupported formatting mode: {mode}")
            return ""

        logging.debug(f"Sending prompt to provider '{self.provider_name}' with mode '{mode}'.")
        formatted_output = current_provider.format_text(prompt=prompt, stop_sequence=stop_sequence)

        if not formatted_output:
            logging.error("Formatting failed or returned empty output.")
            return ""

        return formatted_output