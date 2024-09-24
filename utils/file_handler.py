# utils/file_handler.py

import yaml
import json
import logging
from typing import List, Dict, Any, Optional, Union
import re
import os
import pandas as pd
import tempfile
import zipfile
import subprocess
from docx import Document
from docxcompose.composer import Composer
from dotenv import load_dotenv
from utils.validation import mask_api_key
from utils.record import Record


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



def doc_to_docx_pipeline(input_doc_path, output_docx_path):

    """Pipeline to convert .doc to .docx and append OLE content."""

    # Step 1: Convert the original .doc file to .docx
    main_docx_content = doc_to_docx(input_doc_path)
    if main_docx_content is None:
        print(f"Failed to convert the original .doc file: {input_doc_path}")
        return

    # Save the converted main .docx to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_main_docx:
        tmp_main_docx.write(main_docx_content)
        main_docx_path = tmp_main_docx.name

    # Step 2: Extract and process OLE objects
    ole_objects = extract_ole_objects(main_docx_path)

    # Convert and collect embedded .doc content to append
    appended_docs = []
    for i, ole_content in enumerate(ole_objects):
        # Convert the extracted .doc or .zip containing .doc files
        extracted_docs = process_ole_content(ole_content)
        for extracted_doc_content in extracted_docs:
            appended_docs.append(extracted_doc_content)

    # Step 3: Append the extracted contents to the main document
    if appended_docs:
        combined_doc = append_documents(main_docx_path, appended_docs)
        # Save the combined document to the specified output path
        combined_doc.save(output_docx_path)
        print(f"Combined document saved at: {output_docx_path}")
    else:
        print("No content to append.")
        # Save the original .docx content without changes
        with open(output_docx_path, 'wb') as f:
            f.write(main_docx_content)

def doc_to_docx(input_doc_path):

    """Convert a .doc file to .docx using LibreOffice."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Copy the input .doc file to the temporary directory
        doc_path = os.path.join(tmpdirname, 'input.doc')
        with open(doc_path, 'wb') as doc_file:
            with open(input_doc_path, 'rb') as f:
                doc_file.write(f.read())

        # Convert the .doc file to .docx using LibreOffice
        try:
            subprocess.run(['soffice', '--headless', '--convert-to', 'docx', doc_path, '--outdir', tmpdirname],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"Error during conversion: {e}")
            return None

        # Read the converted .docx content
        docx_path = os.path.join(tmpdirname, 'input.docx')
        if os.path.exists(docx_path):
            with open(docx_path, 'rb') as docx_file:
                return docx_file.read()
        else:
            print("Conversion failed: .docx file not found.")
            return None

def extract_ole_objects(docx_path):

    """Extract OLE objects from a .docx file."""
    ole_objects = []
    with zipfile.ZipFile(docx_path, 'r') as docx:
        # Locate embedded OLE objects
        for item in docx.namelist():
            if item.startswith('word/embeddings'):
                # Extract the OLE object file
                with docx.open(item) as file:
                    ole_objects.append(file.read())
    return ole_objects

def process_ole_content(ole_content):

    """Process OLE content, handle both .doc files and .zip containing .doc files."""

    processed_docs = []

    with tempfile.TemporaryDirectory() as tmpdirname:
        # Save the OLE content to a temporary file
        ole_path = os.path.join(tmpdirname, 'embedded_object')
        with open(ole_path, 'wb') as ole_file:
            ole_file.write(ole_content)

        # Try to handle OLE content as a .doc file directly
        try:
            # Convert the binary content directly using the helper function
            docx_content = doc_to_docx(ole_path)
            if docx_content:
                processed_docs.append(docx_content)
        except Exception as e:
            print(f"Failed to convert as a .doc file: {e}")

        # If the above fails, try to handle OLE content as a .zip file containing .doc files
        if not processed_docs and zipfile.is_zipfile(ole_path):
            try:
                with zipfile.ZipFile(ole_path, 'r') as zip_file:
                    for item in zip_file.namelist():
                        if item.endswith('.doc'):
                            with zip_file.open(item) as doc_file:
                                doc_content = doc_file.read()
                                # Save the content to a temporary .doc file
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.doc') as temp_doc:
                                    temp_doc.write(doc_content)
                                    temp_doc.flush()
                                    # Convert the extracted .doc to .docx
                                    docx_content = doc_to_docx(temp_doc.name)
                                    if docx_content:
                                        processed_docs.append(docx_content)
            except Exception as e:
                print(f"Failed to process the OLE as a .zip file: {e}")

    return processed_docs


def append_documents(main_doc_path, appended_docs):
    """Append documents to the main document and return the combined document."""
    
    # Load the main document

    main_doc = Document(main_doc_path)
    composer = Composer(main_doc)

    # Append each document to the main document
    for doc_content in appended_docs:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_docx:
            tmp_docx.write(doc_content)
            tmp_docx.flush()
            tmp_doc = Document(tmp_docx.name)
            composer.append(tmp_doc)

    return main_doc



def create_documents_dataframe(base_folder):
    """
    Traverse the folder structure and create a dataframe containing information
    about legal documents, including their hierarchy and related documents.

    Args:
    - base_folder: The root directory containing the documents, categorized by type.

    Returns:
    - A pandas DataFrame with columns: 'Category', 'Document ID', 'Hierarchy Level', 'Document Path', 'Parent Document ID'
    """
    documents_data = []

    # Traverse the base folder
    for category in os.listdir(base_folder):
        category_path = os.path.join(base_folder, category)

        if os.path.isdir(category_path):
            # Traverse files and subdirectories within the category
            for root, dirs, files in os.walk(category_path):
                for file in files:
                    # Extract the hierarchy identifier (e.g., "1.1.1") and the rest of the document name
                    file_parts = file.split(' ', 1)
                    if len(file_parts) < 2:
                        continue  # Skip files that don't match the expected naming convention

                    hierarchy_id = file_parts[0]  # "1.1.1" part
                    doc_name = file_parts[1]  # Remaining part of the document name
                    doc_id = hierarchy_id  # Document ID is the hierarchy identifier

                    # Determine hierarchy level based on the hierarchy_id
                    hierarchy_level = len(hierarchy_id.split('.'))  # e.g., "1.1.1" has 3 dots -> level 3

                    # Determine the parent document ID
                    parent_doc_id = None
                    if hierarchy_level > 1:  # Only look for a parent if it's not the top level
                        parent_hierarchy_id = '.'.join(hierarchy_id.split('.')[:-1])  # Remove the last part
                        parent_doc_id = parent_hierarchy_id  # Parent is the preceding identifier

                    # Add document info to the list
                    documents_data.append({
                        'Category': category,
                        'Document ID': doc_id,
                        'Document Name': doc_name,
                        'Hierarchy Level': hierarchy_level,
                        'Document Path': os.path.join(root, file),
                        'Parent Document ID': parent_doc_id
                    })

    # Create a DataFrame
    df = pd.DataFrame(documents_data)
    return df