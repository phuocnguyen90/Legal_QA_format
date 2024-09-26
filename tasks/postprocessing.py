# post_processing.py

import logging
from tqdm import tqdm
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.file_handler import read_input_file, split_into_records, save_processed_record, append_to_output_file, load_record
from utils.validation import load_schema, validate_record
from utils.validation import mask_api_key
from utils.load_config import load_config
from providers import ProviderFactory  # Ensure providers are properly structured
from utils.retry_handler import retry
from utils.record import Record, parse_record

# logging.getLogger(__name__)

def main_postprocessing():
    # Load configuration
    config = load_config('config/config.yaml')
    
    # Prepare a copy of the config for logging with masked API keys
    log_config = config.copy()
    providers = ['groq', 'google_gemini', 'ollama', 'openai']
    for provider in providers:
        if provider in log_config and 'api_key' in log_config[provider]:
            original_key = log_config[provider]['api_key']
            masked_key = mask_api_key(original_key)
            log_config[provider]['api_key'] = masked_key
    
    # Log the masked configuration
    logging.info(f"Loaded configuration: {log_config}")
    
    # Define input and output file paths
    input_file = config['processing']['processed_file']  # Output from preprocessing
    output_file = config['processing']['final_output_file']  # Define in config.yaml
    
    # Ensure the output file is empty before starting
    open(output_file, 'w', encoding='utf-8').close() 
    logging.info(f"Output file {output_file} initialized.")
    
    # Step 1: Read data from the input file
    try:
        data = read_input_file(input_file)
    except Exception as e:
        logging.error(f"Failed to read input file: {e}")
        print(f"Failed to read input file: {e}")
        return
    
    # Step 2: Split data into records
    records_raw = data.strip().split('\n\n')  # Assuming records are separated by double newlines
    
    # Verify if records were split correctly
    if not records_raw:
        logging.error("No records found in processed data. Please check the preprocessing step.")
        print("No records found in processed data. Please check the preprocessing step.")
        return
    
    # Step 3: Load postprocessing schema
    try:
        postprocessing_schema = load_schema('config/schemas/postprocessing_schema.yaml')
    except Exception as e:
        logging.error(f"Failed to load postprocessing schema: {e}")
        print(f"Failed to load postprocessing schema: {e}")
        return
    
    # Step 4: Process each record
    total_records = len(records_raw)
    
    for idx, raw_record in enumerate(tqdm(records_raw, desc="Postprocessing Records"), start=1):
        logging.info(f"Postprocessing record {idx}/{total_records}")
        print(f"Postprocessing record {idx}/{total_records}")
        
        # Load the Record object
        record = load_record(raw_record, ProviderFactory)
        if not record:
            logging.warning(f"Record {idx} could not be loaded and will be skipped.")
            print(f"Record {idx} could not be loaded and will be skipped.")
            continue
        
        # Postprocessing
        postprocessed_record = postprocess_record(record)
        
        # Validate Postprocessed Data
        if not validate_record(postprocessed_record.to_dict(), postprocessing_schema):
            logging.warning(f"Postprocessed record ID {postprocessed_record.id} failed validation.")
            print(f"Postprocessed record ID {postprocessed_record.id} failed validation.")
            continue
        
        # Save postprocessed record
        try:
            append_to_output_file(output_file, postprocessed_record.to_dict())
            logging.info(f"Record ID {postprocessed_record.id} postprocessed and saved successfully.")
        except Exception as e:
            logging.error(f"Failed to save postprocessed record ID {postprocessed_record.id}: {e}")
            print(f"Failed to save postprocessed record ID {postprocessed_record.id}: {e}")
            continue
        
    print(f"Postprocessing complete. Final data saved to {output_file}")
    logging.info("All records postprocessed successfully.")

def postprocess_record(record):
    """
    Apply postprocessing steps to the record.
    
    :param record: Dictionary containing processed record data.
    :return: Postprocessed record dictionary.
    """
    try:
        # Example postprocessing steps:
        
        # Check for factual inconsistencies (Placeholder logic)
        # In reality, you'd use an LLM or another service to perform factual checks
        # Here, we'll assume all records are factually correct
        
        # Example: Add a timestamp
        record['processed_timestamp'] = "2024-09-22T16:20:00Z"  # Replace with actual timestamp
        
        return record
    except Exception as e:
        logging.error(f"Error postprocessing record ID {record.get('id')}: {e}")
        return record  # Return the record as-is if postprocessing fails