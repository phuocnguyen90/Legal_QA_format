# utils/file_handler.py

import re
import json
import logging

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
    Split the raw data into individual records using <id=number></id=number> tags.
    """
    try:
        pattern = r'(<id=\d+>.*?</id=\d+>)'
        records = re.findall(pattern, data, re.DOTALL)
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

def append_to_output_file(file_path, record):
    """
    Append a single processed record to the output file in JSON format.
    """
    try:
        with open(file_path, 'a', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
            f.write('\n\n')
        logging.info(f"Appended processed record ID {record['id']} to '{file_path}'.")
    except Exception as e:
        logging.error(f"Error appending to output file '{file_path}': {e}")
