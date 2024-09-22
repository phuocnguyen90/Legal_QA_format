import os
import re
import logging
from groq import Groq
from tqdm import tqdm
import time
import json
import yaml
from utils import *

# ------------------------------ Configuration ------------------------------ #

def load_config(config_path='config.yaml'):
    """
    Load the YAML configuration file and resolve environment variables.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        # Resolve environment variables in config
        for provider in ['groq', 'google_gemini', 'openai']:
            if provider in config:
                for key, value in config[provider].items():
                    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                        env_var = value[2:-1]
                        config[provider][key] = os.environ.get(env_var, "")
        
        logging.info(f"Configuration loaded from '{config_path}'.")
        return config
    except Exception as e:
        logging.error(f"Error loading configuration file '{config_path}': {e}")
        raise


initial_requirements = """
- Do not add any comment. Just produce the output as required below. Preserve Vietnamese contents inside the tags. Do not attempt to translate any content.
- Remove any content that contains Personally Identifiable Information (PII).
- Remove any part of the text that is not directly related to the title, particularly remove all promotional texts.
- If possible, rephrase the title inside the <title> tag so that it better generalizes the content inside the <content> tag.
- If possible, remove all irrelevant, incoherent, or badly formatted texts inside the <content> tag.
"""

client = Groq(api_key=GROQ_API_KEY)

def main_preprocessing():
    # Define input and output file paths
    input_file = 'input.txt'  # Update the path as needed
    output_file = 'preprocessed.txt'
    
    # Ensure the output file is empty before starting
    open(output_file, 'w', encoding='utf-8').close() 
    logging.info(f"Output file {output_file} initialized.")
    
    # Step 1: Read data from the input file
    data = read_data(input_file)
    
    # Step 2: Split data into records
    records = split_records(data)
    
    # Verify if records were split correctly
    if not records:
        logging.error("No records found. Please check the input file format.")
        print("No records found. Please check the input file format.")
        return
    
    # Step 3: Process each record
    total_records = len(records)
    
    for idx, record in enumerate(tqdm(records, desc="Preprocessing Records"), start=1):
        logging.info(f"Processing record {idx}/{total_records}")
        print(f"Processing record {idx}/{total_records}")
        
        processed_data = process_record(record, initial_requirements, client)
        
        if processed_data:
            save_processed_record(processed_data, output_file)
        else:
            logging.warning(f"Record {idx} was not processed successfully.")
            print(f"Record {idx} was not processed successfully.")
        
        # Optional: Sleep to respect API rate limits
        time.sleep(3)  # Adjust as needed based on API rate limits
    
    print(f"Preprocessing complete. Cleaned data saved to {output_file}")
    logging.info("All records preprocessed successfully.")
