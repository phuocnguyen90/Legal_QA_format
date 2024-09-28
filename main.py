# main.py
import os
import sys
import logging
from utils.load_config import load_config
from utils.logging_setup import setup_logging
from utils.validation import mask_api_key, load_schema, validate_record
from utils.retry_handler import retry
from tasks.preprocessing import Preprocessor
from tasks.postprocessing import main_postprocessing
from utils.input_processor import InputProcessor
from utils.file_handler import output_2_jsonl
def main():
    # Remove all existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create a StreamHandler with UTF-8 encoding
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)  # Set the desired logging level

    # Define a formatter that includes the timestamp, logger name, level, and message
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)

    # Set the encoding to 'utf-8' (available in Python 3.9+)
    # For earlier versions, you might need to use a workaround
 
    # Add the handler to the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set the desired logging level
    logger.addHandler(stream_handler)
    # Load configuration
    try:
        config = load_config('config/config.yaml')
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return


     # Initialize InputProcessor

    try:
        input_processor = InputProcessor(config=config)
    except Exception as e:
        logger.error(f"Failed to initialize InputProcessor: {e}")
        return


    # Initialize Preprocessor
    try:
        preprocessor = Preprocessor(config)
        logger.info("Preprocessor initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Preprocessor: {e}")
        return

    # Define input and output file paths
    # input_file = config['processing']['input_file']
    # input_file="data\\raw\\ND-01-2020.docx"
    input_file="data\\raw\\input.txt"
    output_file = config['processing']['preprocessed_file']


    # Process all records

    try:
        records=input_processor.process_input_file(input_file,return_type='dict',record_type='QA')
        logger.info("All records processed successfully.")
        output_2_jsonl(output_file,records)
    except Exception as e:
        logger.error(f"Error processing records: {e}")

if __name__ == "__main__":
    main()