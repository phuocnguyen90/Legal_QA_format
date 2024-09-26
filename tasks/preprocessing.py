# pre_processing.py

import logging
import time
import json
from typing import List, Optional, Dict, Any

from utils.file_handler import read_input_file, split_into_records, save_processed_record, load_record
from utils.validation import load_schema, validate_record, mask_api_key
from utils.load_config import load_config
from providers import ProviderFactory  
from utils.retry_handler import retry
from utils.record import Record, parse_record
from utils.llm_formatter import LLMFormatter

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Preprocessor:
    """
    A class to handle the pre-processing of records.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Preprocessor with the given configuration.

        :param config: Configuration dictionary.
        """
        # Load configuration
        self.config = config
        logger.info("Preprocessor initialized with provided configuration.")

        # Initialize the provider
        self.provider = self.initialize_provider()

    def initialize_provider(self) -> Any:
        """
        Initialize the API provider based on the configuration.

        :return: An instance of the API provider.
        """
        provider_name = self.config.get('provider', '').strip()
        if not provider_name:
            logger.error("API provider not specified in the configuration.")
            raise ValueError("API provider not specified in the configuration.")

        provider_config = self.config.get(provider_name, {})
        if not provider_config:
            logger.error(f"No configuration found for provider '{provider_name}'.")
            raise ValueError(f"No configuration found for provider '{provider_name}'.")

        # Retrieve the preprocessing schema path from config
        try:
            schema_path = self.config['processing']['schema_paths']['pre_processing_schema']
            preprocessing_schema = load_schema(schema_path)
            requirements = preprocessing_schema.get('pre_process_requirements', [])
            if not requirements:
                logger.error("No 'pre_process_requirements' found in the schema.")
                raise ValueError("No 'pre_process_requirements' found in the schema.")
            logger.info(f"Loaded pre_process_requirements from '{schema_path}'.")
        except KeyError as ke:
            logger.error(f"Missing key in configuration: {ke}")
            raise
        except Exception as e:
            logger.error(f"Failed to load preprocessing schema: {e}")
            raise

        try:
            provider = ProviderFactory.get_provider(provider_name, provider_config, requirements)
            logger.info(f"Initialized provider: {provider_name}")
            return provider
        except Exception as e:
            logger.error(f"Failed to initialize provider '{provider_name}': {e}")
            raise

    def preprocess_record(self, record: Record) -> Record:
        """
        Apply preprocessing steps to a Record.

        :param record: The Record instance to preprocess.
        :return: The preprocessed Record instance.
        """
        # Placeholder for actual preprocessing logic that can be handled locally
        # For example, removing PII, cleaning text, etc.
        # Implement the actual preprocessing as needed
        # Example:
        # record.content = clean_text(record.content)
        return record  # Return the record as-is for now

    def process_record(self, record: Record) -> Optional[Record]:
        """
        Process a single Record instance.

        :param record: The Record instance to process.
        :return: The processed Record instance or None if processing fails.
        """
        try:
            # Preprocessing (e.g., cleaning, formatting)
            processed_record = self.preprocess_record(record)
            logger.debug(f"Record ID {processed_record.id} preprocessed.")

            # Validate Preprocessed Data
            schema_path = 'config/schemas/preprocessing_schema.yaml'
            is_valid = validate_record(
                record=processed_record.to_dict(),
                schema_path=schema_path,
                mode="preprocessing",
                config=self.config
            )
            if not is_valid:
                logger.warning(f"Preprocessed record ID {processed_record.id} failed validation.")
                return None
            logger.debug(f"Record ID {processed_record.id} passed validation.")

            # Process with API
            processed_data = self.provider.process_record(processed_record.to_dict())

            if not processed_data:
                logger.warning(f"Record ID {processed_record.id} was not processed successfully by {self.config['provider']}.")
                return None

            # Initialize a new Record object from the processed JSON
            try:
                processed_record_json = json.loads(processed_data)
                processed_record_new = Record.from_dict(processed_record_json)
                if not processed_record_new:
                    logger.error(f"Processed data for record ID {processed_record.id} is invalid.")
                    return None
                logger.debug(f"Record ID {processed_record_new.id} processed successfully.")
            except json.JSONDecodeError as e:
                logger.error(f"JSON decoding error for record ID {processed_record.id}: {e}")
                return None

            return processed_record_new

        except Exception as e:
            logger.error(f"Failed to process record ID {record.id}: {e}")
            return None

    def process_all_records(self, input_file: str, output_file: str, formatted: bool = True):
        """
        Process all records from the input file and save them to the output file.

        :param input_file: Path to the input file containing raw records.
        :param output_file: Path to the output file to save processed records.
        :param formatted: Boolean indicating if the input records are formatted.
        """
        try:
            # Ensure the output file is empty before starting
            open(output_file, 'w', encoding='utf-8').close()
            logger.info(f"Output file {output_file} initialized.")

            # Read data from the input file
            data = read_input_file(input_file)

            # Split data into individual records
            records_raw = split_into_records(data)

            if not records_raw:
                logger.error("No records found. Please check the input file format.")
                print("No records found. Please check the input file format.")
                return

            # Iterate through each raw record and process
            total_records = len(records_raw)
            for idx, raw_record in enumerate(records_raw, start=1):
                logger.info(f"Processing record {idx}/{total_records}")
                print(f"Processing record {idx}/{total_records}")

                # Load the Record object
                record = self.load_record(raw_record, formatted)

                if not record:
                    logger.warning(f"Record {idx} could not be loaded and will be skipped.")
                    print(f"Record {idx} could not be loaded and will be skipped.")
                    continue

                # Process the record
                processed_record = self.process_record(record)

                if not processed_record:
                    logger.warning(f"Record ID {record.id} failed processing and will be skipped.")
                    print(f"Record ID {record.id} failed processing and will be skipped.")
                    continue

                # Save the processed record
                save_processed_record(processed_record.to_dict(), output_file)
                logger.info(f"Record ID {processed_record.id} processed and saved successfully.")

                # Optional: Sleep to respect API rate limits
                time.sleep(self.config['processing'].get('delay_between_requests', 1))

            print(f"Preprocessing complete. Cleaned data saved to {output_file}")
            logger.info("All records preprocessed successfully.")

        except Exception as e:
            logger.error(f"An error occurred while processing all records: {e}")
            print(f"An error occurred while processing all records: {e}")

    def load_record(self, raw_input: str, formatted: bool = True) -> Optional[Record]:
        """
        Load a Record object from raw input, distinguishing between formatted and unformatted.

        :param raw_input: The raw input string (tagged text, JSON, or unformatted text).
        :param formatted: Boolean indicating if the input is formatted.
        :return: Record object or None if loading fails.
        """
        try:
            if formatted:
                # Attempt to parse as JSON
                record = Record.from_json(raw_input)
                if record:
                    logger.debug("Record loaded from JSON.")
                    return record

                # Attempt to parse as tagged text
                record = Record.from_tagged_text(raw_input)
                if record:
                    logger.debug("Record loaded from tagged text.")
                    return record

                # If formatted is True but parsing fails, log error
                logger.error("Failed to parse formatted record. Ensure it is valid JSON or tagged text.")
                return None
            else:
                # Treat as unformatted text
                record = Record.from_unformatted_text(raw_input, LLMFormatter)
                if record:
                    logger.debug("Record loaded from unformatted text via LLM.")
                    return record

                logger.error("Failed to load unformatted record after LLM processing.")
                return None

        except Exception as e:
            logger.error(f"Unexpected error in load_record: {e}")
            return None