# utils/validation.py

import json
import yaml
import logging
from providers.groq_provider import GroqProvider
from jsonschema import validate, ValidationError
from typing import Dict, Any, Optional

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

def validate_record(record, schema, mode='default', config=None):
    """
    Validate a single record against the provided JSON schema or mode-based requirements.
    
    :param record: The record to validate. Can be a dictionary or a JSON string.
    :param schema: The JSON schema to validate against (as a dictionary).
    :param mode: Optional mode for validation ("preprocessing", "postprocessing"). Defaults to 'default'.
    :param config: Configuration dictionary loaded by load_config(). Required if mode is specified.
    :return: True if valid, False otherwise.
    """
    try:
        # Step 1: Determine the validation schema based on mode
        if mode == 'preprocessing':
            if config is None:
                logging.error("Configuration must be provided for preprocessing mode.")
                return False
            schema = config['tasks'].get('pre_process_requirements')
            if schema is None:
                logging.error("pre_process_requirements not found in configuration.")
                return False
            requirements = "pre_process_requirements"
        
        elif mode == 'postprocessing':
            if config is None:
                logging.error("Configuration must be provided for postprocessing mode.")
                return False
            schema = config['processing'].get('post_process_requirements')
            if schema is None:
                logging.error("post_process_requirements not found in configuration.")
                return False
            requirements = "post_process_requirements"
        
        elif mode == 'default':
            # Use the provided schema
            pass
        else:
            logging.error(f"Invalid mode '{mode}' specified for validation.")
            return False
        
        # Step 2: Ensure record is a dictionary
        if not isinstance(record, dict):
            logging.debug("Record is not a dictionary. Attempting to convert from JSON string.")
            try:
                record = json.loads(record)
                logging.debug("Record successfully converted to dictionary with JSON module.")
            except json.JSONDecodeError as jde:
                logging.error(f"Failed to decode JSON for record: {jde.msg}")
                return False
        
        # Step 3: If in preprocessing or postprocessing mode, send to LLM for compliance
        if mode in ['preprocessing', 'postprocessing']:
            logging.debug(f"Sending record ID {record.get('id', 'N/A')} to LLM for compliance check.")
            is_compliant = llm_validate(record, schema)
            if not is_compliant:
                logging.error(f"LLM validation failed for record ID {record.get('id', 'N/A')}.")
                return False
            logging.debug(f"LLM validation passed for record ID {record.get('id', 'N/A')}.")
        
        # Step 4: Validate the record against the schema
        validate(instance=record, schema=requirements)
        
        # Step 5: Extract 'id' for logging, if available
        record_id = record.get('id', 'N/A')
        logging.info(f"Record ID {record_id} passed validation in mode '{mode}'.")
        return True
    
    except ValidationError as ve:
        # Extract 'id' for logging, if available
        record_id = record.get('id', 'N/A') if isinstance(record, dict) else 'N/A'
        logging.error(f"Validation error for record ID {record_id} in mode '{mode}': {ve.message}")
        return False
    except Exception as e:
        # General exception handling
        record_id = record.get('id', 'N/A') if isinstance(record, dict) else 'N/A'
        logging.error(f"Unexpected error during validation for record ID {record_id} in mode '{mode}': {e}")
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
    
def llm_validate(record: Dict[str, Any], requirements: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """
    Validate the record against the requirements using the GroqProvider.

    :param record: The record to validate as a dictionary.
    :param requirements: The JSON schema or requirements to validate against.
    :param config: The configuration dictionary loaded by load_config().
    :return: True if compliant, False otherwise.
    """
    try:
        logging.debug("Initializing GroqProvider for LLM validation.")
        # Initialize GroqProvider with config and requirements
        groq_provider = GroqProvider(config=config, requirements=requirements)
        
        logging.debug("Processing record with GroqProvider.")
        processed_record_json = groq_provider.process_record(record)
        
        if processed_record_json is None:
            logging.error("GroqProvider failed to process the record.")
            return False
        
        # Optionally, you can further process or validate the `processed_record_json` if needed
        # For now, we'll assume that a successful processing indicates compliance
        logging.debug("GroqProvider successfully processed the record.")
        return True

    except Exception as e:
        logging.error(f"LLM validation failed with exception: {e}")
        return False