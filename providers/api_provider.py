# providers/api_provider.py

from abc import ABC, abstractmethod

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
