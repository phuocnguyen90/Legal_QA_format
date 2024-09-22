# utils/validation.py

import json
import logging
from jsonschema import validate, ValidationError

def load_schema(schema_path):
    """
    Load a JSON schema from a file.
    """
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        logging.info(f"Loaded schema from '{schema_path}'.")
        return schema
    except Exception as e:
        logging.error(f"Error loading schema '{schema_path}': {e}")
        raise

def validate_record(record, schema):
    """
    Validate a single record against the provided schema.
    """
    try:
        validate(instance=record, schema=schema)
        return True
    except ValidationError as ve:
        logging.error(f"Validation error for record ID {record.get('id')}: {ve}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during validation: {e}")
        return False
