# providers/provider_factory.py

import logging
from typing import Optional, Dict, Any
from providers.api_provider import APIProvider
from providers.groq_provider import GroqProvider
# Import other providers as needed, e.g., OpenAIProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProviderFactory:
    """
    Factory class to instantiate APIProvider subclasses based on provider name.
    """

    @staticmethod
    def get_provider(provider_name: str, config: Dict[str, Any], requirements: str) -> Optional[APIProvider]:
        """
        Get an instance of the specified APIProvider subclass.

        :param provider_name: Name of the provider (e.g., 'groq', 'openai').
        :param config: Configuration dictionary for the provider.
        :param requirements: Preprocessing requirements as a string.
        :return: An instance of APIProvider or None if provider is unsupported.
        """
        provider_name = provider_name.lower()
        if provider_name == "groq":
            return GroqProvider(config, requirements)
        elif provider_name == "openai":
            from providers.openai_provider import OpenAIProvider
            return OpenAIProvider(config, requirements)
        # Add additional providers here
        else:
            logger.error(f"Unsupported provider: {provider_name}")
            return None
