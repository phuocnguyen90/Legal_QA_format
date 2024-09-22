# providers/__init__.py

from providers.groq_provider import GroqProvider
from providers.google_gemini_provider import GoogleGeminiProvider

class ProviderFactory:
    @staticmethod
    def get_provider(provider_name, config, requirements):
        if provider_name.lower() == "groq":
            return GroqProvider(config, requirements)
        elif provider_name.lower() == "google_gemini":
            return GoogleGeminiProvider(config, requirements)
        else:
            raise ValueError(f"Unsupported provider: {provider_name}")
