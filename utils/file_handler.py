# utils/file_handler.py

import yaml
import json
import logging
from typing import List, Dict, Any, Optional
import re
import os
from dotenv import load_dotenv
from utils.validation import mask_api_key
from utils.record import Record
from utils.llm_processor import format_unformatted_text

# logging.getLogger(__name__)
def load_record(raw_input: str, llm_processor, is_formatted: bool = True) -> Optional[Record]:
    """
    Load a Record object from raw input, determining the format.

    :param raw_input: The raw input string (tagged text, JSON, or unformatted text).
    :param llm_processor: A callable that processes unformatted text.
    :param is_formatted: Boolean indicating if the input is formatted by default.
    :return: Record object or None if loading fails.
    """
    try:
        if is_formatted:
            # Attempt to parse as JSON
            try:
                record = Record.from_json(raw_input)
                if record:
                    logging.debug("Record loaded from JSON.")
                    return record
            except Exception as e:
                logging.debug(f"Failed to load record from JSON: {e}")

            # Attempt to parse as tagged text
            record = Record.from_tagged_text(raw_input)
            if record:
                logging.debug("Record loaded from tagged text.")
                return record

            # If parsing fails, treat as unformatted
            logging.debug("Record could not be parsed as formatted. Treating as unformatted.")
            record = Record.from_unformatted_text(raw_input, llm_processor)
            if record:
                logging.debug("Record loaded from unformatted text via LLM.")
                return record
        else:
            # Treat input as unformatted
            logging.debug("Input specified as unformatted. Processing with LLM.")
            record = Record.from_unformatted_text(raw_input, llm_processor)
            if record:
                logging.debug("Record loaded from unformatted text via LLM.")
                return record

        logging.error("Failed to load record from any supported format.")
        return None

    except Exception as e:
        logging.error(f"Unexpected error in load_record: {e}")
        return None

def read_input_file(file_path):
    """
    Read the raw input file and return its content.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read()
        logging.info(f"Read data from '{file_path}'.")
        return data
    except Exception as e:
        logging.error(f"Error reading input file '{file_path}': {e}")
        raise

def split_into_records(data):
    """
    Split the raw data into individual records using either <id=number></id=number> tags or JSON objects with an "id" property.
    """
    try:
        # Define patterns for tagged text and JSON text
        tag_pattern = r'<id=\d+>.*?</id=\d+>'
        json_pattern = r'\{[^{}]*"id"\s*:\s*\d+[^{}]*\}'
        
        # Combine both patterns using alternation
        combined_pattern = f'({tag_pattern})|({json_pattern})'
        
        # Find all matches in the data
        matches = re.finditer(combined_pattern, data, re.DOTALL)
        
        # Extract matched strings from the iterator
        records = [match.group(0) for match in matches]
        
        logging.info(f"Split data into {len(records)} records.")
        return records
    except Exception as e:
        logging.error(f"Error splitting data into records: {e}")
        return []

def parse_record(record_str):
    """
    Parse a single record string into a dictionary.
    """
    try:
        record = {}
        record['id'] = int(re.search(r'<id=(\d+)>', record_str).group(1))
        record['title'] = re.search(r'<title>(.*?)</title>', record_str, re.DOTALL).group(1).strip()
        record['published_date'] = re.search(r'<published_date>(.*?)</published_date>', record_str, re.DOTALL).group(1).strip()
        categories_str = re.search(r'<categories>(.*?)</categories>', record_str, re.DOTALL).group(1).strip()
        record['categories'] = re.findall(r'<(.*?)>', categories_str)
        content = re.search(r'<content>(.*?)</content>', record_str, re.DOTALL).group(1).strip()
        record['content'] = content
        return record
    except Exception as e:
        logging.error(f"Error parsing record: {e}")
        return None

def write_output_file(file_path, data):
    """
    Write the processed data to the output file in JSON format.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.info(f"Wrote processed data to '{file_path}'.")
    except Exception as e:
        logging.error(f"Error writing to output file '{file_path}': {e}")

def append_to_output_file(file_path: str, record: Dict[str, Any]):
    """
    Append a processed record to the output file.
    If a record with the same 'id' exists, overwrite it.
    If the file does not exist, create it.

    Records are stored as JSON objects separated by double newlines.

    :param file_path: Path to the output file.
    :param record: Dictionary representing the processed record.
    """
    try:
        # Ensure the record has an 'id' field
        record_id = record.get('id')
        if record_id is None:
            logging.error("Record does not contain an 'id' field.")
            return

        if os.path.exists(file_path):
            logging.debug(f"Output file '{file_path}' exists. Reading existing records.")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = f.read()

            # Parse existing records
            if data.strip():
                record_strings = data.strip().split('\n\n')
                try:
                    records = [json.loads(r) for r in record_strings]
                except json.JSONDecodeError as e:
                    logging.error(f"JSON decoding error while reading '{file_path}': {e}")
                    return
            else:
                records = []

            # Check if a record with the same 'id' exists
            existing_record = next((r for r in records if r.get('id') == record_id), None)
            if existing_record:
                # Overwrite the existing record
                logging.info(f"Overwriting existing record with id {record_id}.")
                records = [record if r.get('id') == record_id else r for r in records]
            else:
                # Append the new record
                logging.info(f"Appending new record with id {record_id}.")
                records.append(record)
        else:
            # File does not exist; create it and add the record
            logging.info(f"Output file '{file_path}' does not exist. Creating and adding record with id {record_id}.")
            records = [record]

        # Write all records back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            for rec in records:
                json.dump(rec, f, ensure_ascii=False, indent=2)
                f.write('\n\n')  # Separator between records

        logging.debug(f"Record with id {record_id} has been successfully saved to '{file_path}'.")

    except Exception as e:
        logging.error(f"An error occurred in append_to_output_file: {e}")

def save_processed_record(record: Dict[str, Any], file_path: str):
    """
    Save a single processed record to the output file.
    Overwrites if record with the same 'id' exists, or appends if it does not.
    
    :param record: Dictionary representing the processed record.
    :param file_path: Path to the output file.
    """
    append_to_output_file(file_path, record)
