# main.py

import os
import json
import logging
import yaml
from tqdm import tqdm

from providers import ProviderFactory
from utils.file_handler import (
    read_input_file,
    split_into_records,
    parse_record,
    write_output_file,
    append_to_output_file
)
from utils.validation import load_schema, validate_record
from utils.rate_limiter import RateLimiter
from utils.retry_handler import retry
from tasks.preprocessing import preprocess_record
from tasks.postprocessing import postprocess_record
from utils.logging_setup import setup_logging

def main():
    # Load configuration
    config_path = 'config/config.yaml'
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return

    # Setup logging
    setup_logging(config['processing']['log_file'])

    # Initialize API provider
    provider_name = config['provider']
    provider_config = config.get(provider_name, {})
    try:
        provider = ProviderFactory.get_provider(provider_name, provider_config, config['tasks']['requirements'])
        logging.info(f"Initialized provider: {provider_name}")
    except Exception as e:
        logging.error(f"Failed to initialize provider '{provider_name}': {e}")
        return

    # Read and split input data
    raw_data = read_input_file(config['processing']['input_file'])
    record_strings = split_into_records(raw_data)

    # Load schemas
    preprocessing_schema = load_schema('config/schemas/preprocessing_schema.yaml')
    postprocessing_schema = load_schema('config/schemas/postprocessing_schema.yaml')

    # Initialize rate limiter (example: max 60 calls per minute)
    rate_limiter = RateLimiter(max_calls=60, period=60)

    # Process each record
    for record_str in tqdm(record_strings, desc="Processing Records"):
        record = parse_record(record_str)
        if not record:
            logging.warning("Skipping malformed record.")
            continue

        # Preprocessing
        record = preprocess_record(record)

        # API Processing with retry
        @retry(max_attempts=3, delay=2, backoff=2)
        def api_process():
            rate_limiter.wait()
            return provider.process_record(record)

        try:
            processed_record_str = api_process()
            if not processed_record_str:
                logging.warning(f"Record ID {record['id']} not processed.")
                continue

            processed_record = json.loads(processed_record_str)

            # Postprocessing
            processed_record = postprocess_record(processed_record)

            # Validation
            if validate_record(processed_record, postprocessing_schema):
                # Save processed record
                append_to_output_file(config['processing']['processed_file'], processed_record)
            else:
                logging.warning(f"Processed record ID {processed_record.get('id')} failed validation.")

        except Exception as e:
            logging.error(f"Failed to process record ID {record.get('id')}: {e}")
            continue

    logging.info("All records processed successfully.")

if __name__ == "__main__":
    main()
