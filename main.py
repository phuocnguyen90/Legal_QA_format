# main.py
import os
import logging
from utils.load_config import load_config
from utils.logging_setup import setup_logging
from utils.validation import mask_api_key, load_schema, validate_record
from utils.retry_handler import retry
from tasks.preprocessing import Preprocessor
from tasks.postprocessing import main_postprocessing
from utils.input_processor import InputProcessor
from utils.file_handler import append_to_output_file
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
     # Initialize InputProcessor

    try:
        input_processor = InputProcessor(config=config)
    except Exception as e:
        logging.error(f"Failed to initialize InputProcessor: {e}")
        return


    # Initialize Preprocessor
    try:
        preprocessor = Preprocessor(config)
        logging.info("Preprocessor initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize Preprocessor: {e}")
        return

    # Define input and output file paths
    # input_file = config['processing']['input_file']
    input_file="data\\raw\\ND-01-2020.docx"
    output_file = config['processing']['preprocessed_file']


    # Process all records

    try:
        records=input_processor.process_input_file(input_file,return_type='dict',record_type='QA')
        logging.info("All records processed successfully.")
        append_to_output_file(output_file,records)
    except Exception as e:
        logging.error(f"Error processing records: {e}")

if __name__ == "__main__":
    main()