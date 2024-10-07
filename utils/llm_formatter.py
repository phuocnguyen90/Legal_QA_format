import logging
import json
import yaml
from typing import Optional, Dict, Any, List, Union
from utils.record import Record  # Importing the Record class for usage

# Placeholder for providers and loading configuration
from providers import ProviderFactory  
from providers.api_provider import APIProvider

class LLMProcessor:
    """
    Consolidated LLM Processor to handle formatting, enrichment, input processing,
    and generating Record instances.
    """
    
    def __init__(self, config_path: str, prompts_path: str = "config/schemas/prompts.yaml"):
        # Load configuration and set up logging
        self.config = self._load_config(config_path)
        self.prompts = self._load_prompts(prompts_path)
        self.provider = self._initialize_provider()

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"LLMProcessor initialized with provider '{self.provider.__class__.__name__}'.")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        # Logic to load YAML configuration, similar to `load_config.py`
        # Load and handle environment variables, etc.
        pass

    def _load_prompts(self, prompts_path: str) -> Dict[str, Any]:
        # Logic to load prompts from YAML
        try:
            with open(prompts_path, 'r', encoding='utf-8') as file:
                prompts = yaml.safe_load(file)
            self.logger.info(f"Loaded prompts from '{prompts_path}'.")
            return prompts.get('prompts', {})
        except FileNotFoundError:
            self.logger.error(f"Prompts file '{prompts_path}' not found.")
            raise

    def _initialize_provider(self) -> APIProvider:
        # Initialize the appropriate LLM provider (OpenAI, etc.) based on the config
        provider_name = self.config.get('provider', 'openai').lower()
        provider_config = self.config.get(provider_name, {})
        return ProviderFactory.get_provider(provider_name, provider_config)

    # Consolidated Input Processing and Formatting

    def process_input(self, raw_text: str, mode: str = "tagged", record_type: str = "DOC") -> Optional[Record]:
        """
        Process input text through LLM to create a Record instance.
        :param raw_text: Unformatted text.
        :param mode: The formatting mode - 'tagged', 'json', etc.
        :param record_type: Type of record to be created.
        :return: A Record instance or None.
        """
        formatted_text = self.format_text(raw_text, mode, record_type)
        if formatted_text:
            return Record.from_tagged_text(formatted_text, record_type)
        return None

    def format_text(
        self, 
        raw_text: str, 
        mode: str = "tagged", 
        record_type: Optional[str] = None
    ) -> Optional[str]:
        """
        Format raw text using the specified mode.
        """
        try:
            prompt_template = self.prompts.get('formatting', {}).get(mode, {}).get('prompt')
            if not prompt_template:
                self.logger.error(f"Prompt for mode '{mode}' not found in prompts.")
                return None

            # Replace the placeholders in the prompt
            prompt = prompt_template.format(raw_text=raw_text)
            formatted_output = self.provider.send_message(prompt=prompt)
            if not formatted_output:
                self.logger.error(f"Failed to format text in mode '{mode}'.")
                return None

            return formatted_output
        except Exception as e:
            self.logger.error(f"Error during formatting: {e}")
            return None

    def enrich_text(self, chunk_text: str) -> Optional[Dict[str, Any]]:
        """
        Enrich a chunk of text, extracting main topics, categories, etc.
        :param chunk_text: The raw chunk of text.
        :return: Enriched data dictionary.
        """
        try:
            formatted_output = self.format_text(raw_text=chunk_text, mode="enrichment")
            if not formatted_output:
                self.logger.error("Enrichment failed or returned empty output.")
                return None

            enriched_data = self._parse_llm_response(formatted_output)
            return enriched_data
        except Exception as e:
            self.logger.error(f"Error enriching text: {e}")
            return None

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response into structured enrichment data.
        """
        enriched_data = {}
        lines = response.split('\n')
        for line in lines:
            if line.startswith('Main Topic:'):
                enriched_data['Main Topic'] = line.replace('Main Topic:', '').strip()
            elif line.startswith('Applicability:'):
                enriched_data['Applicability'] = line.replace('Applicability:', '').strip()
            elif line.startswith('Generated Title:'):
                enriched_data['Generated Title'] = line.replace('Generated Title:', '').strip()
            elif line.startswith('Suggested Categories:'):
                categories = line.replace('Suggested Categories:', '').strip()
                enriched_data['Assigned Categories'] = [cat.strip() for cat in categories.split(',')]
        return enriched_data

    def chunk_text(self, content: str) -> List[str]:
        """
        Split a large text into manageable chunks.
        :param content: The full document text.
        :return: List of text chunks.
        """
        pattern = r'Article\s+\d+[\.,]?\s+'  # Adjust the regex pattern based on content
        chunks = re.split(pattern, content)
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def process_document(self, document_path: str) -> List[Record]:
        """
        Read the document, split it into chunks, enrich each chunk, and create Record objects.
        """
        content = self.read_document_content(document_path)
        chunks = self.chunk_text(content)

        records = []
        for idx, chunk in enumerate(chunks, start=1):
            enriched_data = self.enrich_text(chunk)
            if enriched_data:
                record = Record(
                    record_id=f"doc-{idx}",
                    document_id=None,
                    title=enriched_data.get('Generated Title', ''),
                    content=chunk,
                    chunk_id=f"chunk-{idx}",
                    categories=enriched_data.get('Assigned Categories', [])
                )
                records.append(record)
        return records

    def read_document_content(self, file_path: str) -> str:
        """
        Read content from document files (txt, pdf, docx).
        """
        # Handle different file formats - txt, docx, pdf, etc.
        # This is where you can consolidate the file-reading logic
        pass

