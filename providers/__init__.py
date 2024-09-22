# providers/__init__.py

from providers.openai_provider import OpenAIProvider
from providers.groq_provider import GroqProvider
from providers.gemini_provider import GeminiProvider
from providers.ollama_provider import OllamaProvider

class ProviderFactory:
    @staticmethod
    def get_provider(provider_name, config, requirements):
        if provider_name.lower() == "openai":
            return OpenAIProvider(config, requirements)
        elif provider_name.lower() == "groq":
            return GroqProvider(config, requirements)
        elif provider_name.lower() == "gemini":
            return GeminiProvider(config, requirements)
        elif provider_name.lower() == "ollama":
            return OllamaProvider(config, requirements)
        else:
            raise ValueError(f"Unsupported provider: {provider_name}")
