# main.py
import os
import logging
from utils.load_config import load_config
from utils.logging_setup import setup_logging
from utils.validation import mask_api_key, load_schema, validate_record
from utils.retry_handler import retry
from tasks.preprocessing import Preprocessor
from tasks.postprocessing import main_postprocessing

def main():
    # Load configuration
    try:
        config = load_config('config/config.yaml')
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return

    # Setup logging
    try:
        log_file = config['processing']['log_file']
        setup_logging(log_file, level="INFO")  # Change level as needed
    except Exception as e:
        print(f"Failed to set up logging: {e}")
        return


    # Initialize Preprocessor
    try:
        preprocessor = Preprocessor(config)
        logging.info("Preprocessor initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize Preprocessor: {e}")
        return

    # Define input and output file paths
    input_file = config['processing']['input_file']
    output_file = config['processing']['preprocessed_file']
    formatted = config['processing'].get('formatted', True)  # Default to True

    # Process all records

    try:
        preprocessor.process_all_records(input_file, output_file, formatted=formatted)
        logging.info("All records processed successfully.")
    except Exception as e:
        logging.error(f"Error processing records: {e}")

if __name__ == "__main__":
    main()