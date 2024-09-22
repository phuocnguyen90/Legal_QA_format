# utils/validation.py

import json
import yaml
import logging
from jsonschema import validate, ValidationError
# logging.getLogger(__name__)
def load_schema(schema_path):
    """
    Load a YAML schema from a file.
    
    :param schema_path: Path to the YAML schema file.
    :return: Parsed schema as a Python dictionary.
    """
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = yaml.safe_load(f)
        logging.info(f"Loaded schema from '{schema_path}'.")
        return schema
    except Exception as e:
        logging.error(f"Error loading schema '{schema_path}': {e}")
        raise

def validate_record(record, schema):
    """
    Validate a single record against the provided JSON schema.
    
    :param record: The record to validate. Can be a dictionary or a JSON string.
    :param schema: The JSON schema to validate against (as a dictionary).
    :return: True if valid, False otherwise.
    """
    try:
        # Step 1: Ensure record is a dictionary
        if not isinstance(record, dict):
            logging.debug("Record is not a dictionary. Attempting to convert from JSON string.")
            try:
                record = json.loads(record)
                logging.debug("Record successfully converted to dictionary.")
            except json.JSONDecodeError as jde:
                logging.error(f"Failed to decode JSON for record: {jde.msg}")
                return False

        # Step 2: Validate the record against the schema
        validate(instance=record, schema=schema)
        
        # Step 3: Extract 'id' for logging, if available
        record_id = record.get('id', 'N/A')
        logging.info(f"Record ID {record_id} passed validation.")
        return True

    except ValidationError as ve:
        # Extract 'id' for logging, if available
        record_id = record.get('id', 'N/A') if isinstance(record, dict) else 'N/A'
        logging.error(f"Validation error for record ID {record_id}: {ve.message}")
        return False
    except Exception as e:
        # General exception handling
        record_id = record.get('id', 'N/A') if isinstance(record, dict) else 'N/A'
        logging.error(f"Unexpected error during validation for record ID {record_id}: {e}")
        return False

def mask_api_key(api_key):
    """
    Mask the API key by replacing all characters except the last four with '***'.
    
    :param api_key: The original API key as a string.
    :return: Masked API key.
    """
    try:
        if not isinstance(api_key, str):
            raise TypeError("API key must be a string.")
        if len(api_key) <= 4:
            # If API key is too short, mask entirely
            return 'invalid'
        else:
            return '***' + api_key[-4:]
    except Exception as e:
        logging.error(f"Error masking API key: {e}")
        return '***'