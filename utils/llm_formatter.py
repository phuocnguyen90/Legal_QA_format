# utils/llm_formatter.py

import logging
import json
import yaml
from typing import Optional, Dict, Any
from providers import ProviderFactory  
from providers.openai_provider import OpenAIProvider
from providers.groq_provider import GroqProvider
from providers.api_provider import APIProvider
# logging.getLogger(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.prompts = self._load_prompts(prompts_path)
        self.provider = self._initialize_provider()
        logging.info(f"LLMFormatter initialized with provider '{self.provider_name}'.")

    def _load_prompts(self, prompts_path: str) -> Dict[str, Any]:
        """
        Load prompts from the specified YAML file.

        :param prompts_path: Path to the YAML file containing prompts.
        :return: Dictionary of prompts.
        """
        try:
            with open(prompts_path, 'r', encoding='utf-8') as file:
                prompts = yaml.safe_load(file)
            logger.info(f"Loaded prompts from '{prompts_path}'.")
            return prompts.get('prompts', {})
        except FileNotFoundError:
            logger.error(f"Prompts file '{prompts_path}' not found.")
            raise
        except yaml.YAMLError as ye:
            logger.error(f"YAML parsing error in '{prompts_path}': {ye}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading prompts '{prompts_path}': {e}")
            raise

    def _initialize_provider(self) -> APIProvider:
        """
        Initialize the API provider based on the configuration.

        :return: An instance of the API provider.
        """
        provider_name = self.provider_name
        if not provider_name:
            logger.error("API provider not specified in the configuration.")
            raise ValueError("API provider not specified in the configuration.")

        provider_config = self.config.get(provider_name, {})
        if not provider_config:
            logger.error(f"No configuration found for provider '{provider_name}'.")
            raise ValueError(f"No configuration found for provider '{provider_name}'.")

        # Retrieve requirements if any (used by some providers)
        requirements = self.config.get('processing', {}).get('pre_process_requirements', "")
        try:
            provider = ProviderFactory.get_provider(provider_name, provider_config, requirements)
            logger.info(f"Initialized provider: {provider_name}")
            return provider
        except Exception as e:
            logger.error(f"Failed to initialize provider '{provider_name}': {e}")
            raise

    

    def format_text(self, raw_text: str, mode: str = "tagged", provider: Optional[str] = None) -> str:
        """
        Format raw text into structured formats using the specified mode and provider.

        :param raw_text: The raw unformatted text.
        :param mode: The formatting mode ("tagged", "json", or "enrichment"). Defaults to "tagged".
        :param provider: Optional provider override ("openai", "groq", etc.).
        :return: Formatted text as per the specified mode.
        """
        current_provider = self.provider
        current_provider_name = self.provider_name
        if provider:
            # Initialize the specified provider
            temp_formatter = LLMFormatter(config=self.config, prompts_path="config/schemas/prompts.yaml")
            try:
                temp_formatter.provider = temp_formatter._initialize_provider_override(provider)
                current_provider = temp_formatter.provider
                current_provider_name = provider.lower()
                logger.info(f"Switched to provider '{current_provider_name}' for this formatting task.")
            except Exception as e:
                logger.error(f"Failed to initialize provider '{provider}': {e}. Using default provider '{self.provider_name}'.")
                current_provider = self.provider
                current_provider_name = self.provider_name

        if not current_provider:
            logger.error("No valid LLM provider available for formatting.")
            return ""

        # Retrieve the appropriate prompt based on mode
        if mode in self.prompts:
            if mode == "enrichment":
                prompt_template = self.prompts['enrichment']['enrichment_prompt']
                prompt = prompt_template.format(chunk_text=raw_text)
            else:
                prompt_template = self.prompts['formatting'].get(mode)
                if not prompt_template:
                    logger.error(f"No prompt found for formatting mode '{mode}'.")
                    return ""
                if mode == "json":
                    json_schema = json.dumps(self.config.get('processing', {}).get('json_schema', {}), indent=2)
                    prompt = prompt_template.format(raw_text=raw_text, json_schema=json_schema)
                else:
                    prompt = prompt_template.format(raw_text=raw_text)
        else:
            logger.error(f"Unsupported formatting mode: {mode}")
            return ""

        logging.debug(f"Sending prompt to provider '{current_provider_name}' with mode '{mode}'.")
        formatted_output = current_provider.format_text(prompt=prompt)

        if not formatted_output:
            logger.error("Formatting failed or returned empty output.")
            return ""

        return formatted_output
    
    def _initialize_provider_override(self, provider: str) -> APIProvider:
        """
        Initialize a different provider on the fly.

        :param provider: The LLM provider to use ("openai", "groq", etc.).
        :return: An instance of the specified API provider.
        """
        provider = provider.lower()
        provider_config = self.config.get(provider, {})
        if not provider_config:
            logger.error(f"No configuration found for provider '{provider}'.")
            raise ValueError(f"No configuration found for provider '{provider}'.")

        # Retrieve requirements if any (used by some providers)
        requirements = self.config.get('processing', {}).get('pre_process_requirements', "")
        try:
            provider_instance = ProviderFactory.get_provider(provider, provider_config, requirements)
            logger.info(f"Initialized provider: {provider}")
            return provider_instance
        except Exception as e:
            logger.error(f"Failed to initialize provider '{provider}': {e}")
            raise