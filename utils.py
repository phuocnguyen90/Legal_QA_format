
from jsonschema import validate, ValidationError

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
