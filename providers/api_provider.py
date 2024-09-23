# providers/api_provider.py

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

class APIProvider(ABC):
    def __init__(self, config, requirements):
        """
        Initialize the API provider with configuration and processing requirements.
        """
        self.config = config
        self.requirements = requirements

    @abstractmethod
    def process_record(self, record):
        """
        Process the record using the API.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    
    def format_text(self, prompt: str, stop_sequence: Optional[List[str]] = None) -> str:
        """
        Format text based on the provided prompt.

        :param prompt: The prompt to send to the LLM.
        :param stop_sequence: Optional list of stop sequences to terminate the LLM response.
        :return: The formatted text returned by the LLM.
        """
        pass    
