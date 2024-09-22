
import os
import re
import logging
from groq import Groq
from tqdm import tqdm
import time
import json

import getpass
from jsonschema import validate, ValidationError

# Configure logging
logging.basicConfig(
    filename='utils.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def validate_processed_data(processed_data, schema):
    """
    Validates the processed data against the provided schema.
    """
    try:
        data = json.loads(processed_data)
        validate(instance=data, schema=schema)
        return True
    except ValidationError as ve:
        logging.error(f"Schema validation error: {ve}")
        return False
    except json.JSONDecodeError as je:
        logging.error(f"JSON decode error: {je}")
        return False


def read_data(file_path):
    """
    Reads the entire content of the text file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.read()
        logging.info(f"Successfully read data from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
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

def process_record(record, initial_requirements, client):
    """
    Processes a single record using the Groq API based on initial requirements.
    """
    prompt = f"""Please process the following data according to these requirements:

{initial_requirements}

Here is the data:

{record}

Please provide the processed data in the same format, ensuring that all modifications adhere to the requirements."""
    
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,  # Adjust based on the size of your data and model limits
            temperature=0.7
        )
        
        processed_data = response.choices[0].message.content.strip()
        return processed_data
    except Exception as e:
        logging.error(f"Error processing record: {e}")
        return None
    
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
