# main.py

import os
import re
import logging
import yaml
import json
import time
from tqdm import tqdm
from providers import ProviderFactory

# ------------------------------ Configuration ------------------------------ #

def load_config(config_path='config.yaml'):
    """
    Load the YAML configuration file.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        logging.info(f"Configuration loaded from '{config_path}'.")
        return config
    except Exception as e:
        logging.error(f"Error loading configuration file '{config_path}': {e}")
        raise

# ------------------------------ Helper Functions ------------------------------ #

def read_data(file_path):
    """
    Reads the entire content of the text file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.read()
        logging.info(f"Successfully read data from '{file_path}'.")
        return data
    except Exception as e:
        logging.error(f"Error reading file '{file_path}': {e}")
        raise

def split_records(data):
    """
    Splits the data into individual records using the <id=number></id=number> tags.
    """
    # Regular expression pattern to match each record
    pattern = r'(<id=\d+>.*?</id=\d+>)'
    
    # Find all matches with re.DOTALL to include newlines
    records = re.findall(pattern, data, re.DOTALL)
    
    logging.info(f"Total records found: {len(records)}")
    return records

def save_processed_record(processed_record, output_file):
    """
    Appends the processed record to the output file.
    """
    try:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(processed_record + '\n\n')
        logging.info("Processed record saved successfully.")
    except Exception as e:
        logging.error(f"Error saving processed record: {e}")

# ------------------------------ Main Function ------------------------------ #

def main():
    # Load configuration
    try:
        config = load_config('config.yaml')
    except Exception:
        print("Failed to load configuration. Check 'processing.log' for details.")
        return

    provider_name = config.get('provider', '').strip()
    if not provider_name:
        logging.error("API provider not specified in the configuration.")
        print("API provider not specified in the configuration.")
        return

    # Select the API provider configuration
    provider_config = config.get(provider_name, {})
    if not provider_config:
        logging.error(f"No configuration found for provider '{provider_name}'.")
        print(f"No configuration found for provider '{provider_name}'.")
        return

    # Define processing requirements
    requirements = config.get('processing_requirements', """
- Remove any content that contains Personally Identifiable Information (PII).
- Remove any part of the text that is not directly related to the title, particularly remove all promotional texts.
- If possible, rephrase the title inside the <title> tag so that it better generalizes the content inside the <content> tag.
- If possible, remove all irrelevant, incoherent, or badly formatted texts inside the <content> tag.
""")

    # Initialize logging
    logging.basicConfig(
        filename=config['processing']['log_file'],
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    # Initialize the API provider using the factory
    try:
        provider = ProviderFactory.get_provider(provider_name, provider_config, requirements)
    except Exception as e:
        logging.error(f"Error initializing provider '{provider_name}': {e}")
        print(f"Error initializing provider '{provider_name}'. Check 'processing.log' for details.")
        return

    # Read input data
    input_file = config['processing']['input_file']
    output_file = config['processing']['output_file']
    delay_between_requests = config['processing'].get('delay_between_requests', 1)

    try:
        data = read_data(input_file)
    except Exception:
        print("Failed to read input data. Check 'processing.log' for details.")
        return

    # Split data into records
    records = split_records(data)

    # Verify if records were split correctly
    if not records:
        logging.error("No records found. Please check the input file format.")
        print("No records found. Please check the input file format.")
        return

    # Clear the output file before processing
    try:
        open(output_file, 'w', encoding='utf-8').close()
        logging.info(f"Cleared existing '{output_file}'.")
    except Exception as e:
        logging.error(f"Error clearing '{output_file}': {e}")
        print(f"Error clearing '{output_file}'. Check 'processing.log' for details.")
        return

    # Process each record
    total_records = len(records)
    logging.info(f"Beginning processing of {total_records} records.")
    print(f"Beginning processing of {total_records} records.")

    for idx, record in enumerate(tqdm(records, desc="Processing Records"), start=1):
        logging.info(f"Processing record {idx}/{total_records}.")
        print(f"Processing record {idx}/{total_records}.")

        # Process the record using the selected API provider
        processed_data = provider.process_record(record)

        if processed_data:
            # Save the processed record to the output file
            save_processed_record(processed_data, output_file)
        else:
            logging.warning(f"Record {idx} was not processed successfully.")
            print(f"Record {idx} was not processed successfully.")

        # Optional: Sleep to respect API rate limits
        time.sleep(delay_between_requests)

    print(f"Processing complete. Processed data saved to '{output_file}'.")
    logging.info("All records processed successfully.")

# ------------------------------ Entry Point ------------------------------ #

if __name__ == "__main__":
    main()
