# input_processor.py

import logging
from typing import List, Optional, Union, Dict, Any
import os
import json

import pandas as pd
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from lxml import etree

from tasks.preprocessing import Preprocessor  
from utils.record import Record  
from utils.llm_formatter import LLMFormatter, detect_text_type  

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InputProcessor:
    """
    A class to handle the processing of input files containing records in various formats.
    """

    SUPPORTED_TEXT_EXTENSIONS = ['.txt', '.htm', '.html']
    SUPPORTED_TABULAR_EXTENSIONS = ['.csv', '.xls', '.xlsx']
    SUPPORTED_DOCUMENT_EXTENSIONS = ['.pdf', '.doc', '.docx']


    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the InputProcessor with the given configuration.

        :param config: Configuration dictionary.
        """
        self.config = config
        self.preprocessor = self._initialize_preprocessor()
        logger.info("InputProcessor initialized with provided configuration.")

    def _initialize_preprocessor(self) -> Preprocessor:
        """
        Initialize the Preprocessor with the given configuration.

        :return: An instance of Preprocessor.
        """
        try:
            preprocessor = Preprocessor(config=self.config)
            return preprocessor
        except Exception as e:
            logger.error(f"Failed to initialize Preprocessor: {e}")
            raise

    def process_input_file(
        self,
        file_path: str,
        return_type: str = "record",
        record_type: str = "DOC"
    ) -> List[Union[Record, Dict[str, Any], str]]:
        """
        Process the input file and return the processed records.

        :param file_path: Path to the input file.
        :param return_type: The desired return type for each record ('record', 'dict', 'json').
        :param record_type: Type of the record ('QA' or 'DOC') to prefix the record_id accordingly.
        :return: A list of processed records in the specified return_type.
        """
        if not os.path.isfile(file_path):
            logger.error(f"The file '{file_path}' does not exist.")
            return []

        # Determine the file extension
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()
        logger.info(f"Processing file '{file_path}' with extension '{file_extension}'.")

        processed_records = []

        try:
            if file_extension in self.SUPPORTED_TEXT_EXTENSIONS:
                content = self._extract_text_file(file_path, file_extension)
                text_type = detect_text_type(content)
                logger.debug(f"Detected text type: {text_type}")

                if text_type in ["json", "tagged"]:
                    # Handle JSON and Tagged Text
                    logger.info(f"Processing content as '{text_type}'.")
                    record = Record.parse_record(
                        record_str=content,
                        return_type=return_type,
                        record_type=record_type,
                        llm_formatter=None  # No need for LLMFormatter in this case
                    )
                    if record:
                        processed_records.append(record)
                    else:
                        logger.warning(f"Failed to parse record from '{text_type}' content.")
                elif text_type == "unformatted":
                    # Handle Unformatted Text: Chunk and parse
                    logger.info("Processing content as 'unformatted' text. Initiating chunking.")
                    chunks = self._chunk_text(content)
                    logger.info(f"Created {len(chunks)} chunks from unformatted text.")

                    for idx, chunk in enumerate(chunks, start=1):
                        logger.debug(f"Processing chunk {idx}/{len(chunks)}.")
                        record = Record.parse_record(
                            record_str=chunk,
                            return_type=return_type,
                            record_type=record_type,
                            llm_formatter=self.preprocessor.llm_formatter
                        )
                        if record:
                            processed_records.append(record)
                        else:
                            logger.warning(f"Failed to parse chunk {idx} into a Record.")
                else:
                    logger.error(f"Unsupported text type detected: {text_type}")

            elif file_extension in self.SUPPORTED_TABULAR_EXTENSIONS:
                records = self._process_tabular_file(file_path)
                logger.info(f"Processed {len(records)} records from tabular file.")

                for record in records:
                    parsed_record = Record.parse_record(
                        record_str=json.dumps(record),
                        return_type=return_type,
                        record_type=record_type,
                        llm_formatter=None  # Assuming tabular data is already structured
                    )
                    if parsed_record:
                        processed_records.append(parsed_record)
                    else:
                        logger.warning(f"Failed to parse tabular record with ID: {record.get('record_id') or record.get('id')}")

            elif file_extension in self.SUPPORTED_DOCUMENT_EXTENSIONS:
                content = self._extract_document_file(file_path, file_extension)
                # Treat the extracted content as unformatted text
                text_type = "unformatted"
                logger.debug(f"Detected text type: {text_type}")

                if text_type == "unformatted":
                    logger.info("Processing content as 'unformatted' text. Initiating chunking.")
                    chunks = self._chunk_text(content)
                    logger.info(f"Created {len(chunks)} chunks from document text.")

                    for idx, chunk in enumerate(chunks, start=1):
                        logger.debug(f"Processing chunk {idx}/{len(chunks)}.")
                        record = Record.parse_record(
                            record_str=chunk,
                            return_type=return_type,
                            record_type=record_type,
                            llm_formatter=self.preprocessor.llm_formatter
                        )
                        if record:
                            processed_records.append(record)
                        else:
                            logger.warning(f"Failed to parse chunk {idx} into a Record.")

            else:
                logger.error(f"Unsupported file extension: '{file_extension}'. Supported extensions are: "
                             f"{self.SUPPORTED_TEXT_EXTENSIONS + self.SUPPORTED_TABULAR_EXTENSIONS + self.SUPPORTED_DOCUMENT_EXTENSIONS}")
                return []

        except Exception as e:
            logger.error(f"An error occurred while processing the file '{file_path}': {e}")

        logger.info(f"Processing complete. Total records processed: {len(processed_records)}.")
        return processed_records
    
    def _extract_text_file(self, file_path: str, file_extension: str) -> str:
        """
        Extract text from a text or HTML file.

        :param file_path: Path to the file.
        :param file_extension: File extension.
        :return: Extracted text content.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Extracted content from text file '{file_path}'.")
            if file_extension in ['.htm', '.html']:
                # Parse HTML content and extract text
                parser = etree.HTMLParser()
                tree = etree.parse(file_path, parser)
                content = ' '.join(tree.xpath('//text()'))
                logger.info(f"Extracted text from HTML file '{file_path}'.")
            return content
        except Exception as e:
            logger.error(f"Failed to extract text from file '{file_path}': {e}")
            raise

    def _extract_document_file(self, file_path: str, file_extension: str) -> str:
        """
        Extract text from a PDF or Word document.

        :param file_path: Path to the file.
        :param file_extension: File extension.
        :return: Extracted text content.
        """
        try:
            if file_extension == '.pdf':
                reader = PdfReader(file_path)
                content = ""
                for page in reader.pages:
                    content += page.extract_text() or ""
                logger.info(f"Extracted text from PDF file '{file_path}'.")
                return content
            elif file_extension in ['.doc', '.docx']:
                doc = DocxDocument(file_path)
                content = "\n\n".join([para.text for para in doc.paragraphs])
                logger.info(f"Extracted text from Word document '{file_path}'.")
                return content
            else:
                logger.error(f"Unsupported document extension: '{file_extension}'.")
                return ""
        except Exception as e:
            logger.error(f"Failed to extract text from document file '{file_path}': {e}")
            raise

    def _process_tabular_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a tabular file and convert each row into a dictionary.

        :param file_path: Path to the tabular file.
        :return: List of dictionaries representing records.
        """
        try:
            _, file_extension = os.path.splitext(file_path)
            file_extension = file_extension.lower()

            if file_extension == '.csv':
                df = pd.read_csv(file_path)
            elif file_extension in ['.xls', '.xlsx']:
                df = pd.read_excel(file_path)
            else:
                logger.error(f"Unsupported tabular file extension: '{file_extension}'.")
                return []

            records = df.to_dict(orient='records')
            logger.info(f"Converted tabular file '{file_path}' to {len(records)} records.")
            return records

        except Exception as e:
            logger.error(f"Failed to process tabular file '{file_path}': {e}")
            return []

    def _chunk_text(self, text: str, max_words: int = 300) -> List[str]:
        """
        Chunk the unformatted text into smaller parts based on the max_words threshold.

        :param text: The unformatted input text.
        :param max_words: Maximum number of words per chunk.
        :return: A list of text chunks.
        """
        paragraphs = text.split('\n\n')  # Split by double newlines to get paragraphs
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            word_count = len(paragraph.split())
            if word_count == 0:
                continue  # Skip empty paragraphs

            if len(current_chunk.split()) + word_count <= max_words:
                current_chunk += f"\n\n{paragraph}" if current_chunk else paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph  # Start a new chunk

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks
