# utils/record.py

import json
import logging
from typing import Any, Dict, List, Optional
import re
# logging.getLogger(__name__)
class Record:
    """
    A class to represent and handle a single record.
    """

    def __init__(self, id: int, title: str, published_date: str, categories: List[str], content: str):
        self.id = id
        self.title = title
        self.published_date = published_date
        self.categories = categories
        self.content = content

    @classmethod
    def from_tagged_text(cls, text: str) -> Optional['Record']:
        """
        Initialize a Record object from tagged text.
        Example of tagged text:
        
        <id=1>
        <title>Sample Title</title>
        <published_date>2024-09-22</published_date>
        <categories><Category1><Category2></categories>
        <content>
        Sample content here.
        </content>
        </id=1>
        
        :param text: The raw tagged text string.
        :return: Record object or None if parsing fails.
        """
        try:
            id_match = re.search(r'<id=(\d+)>', text)
            title_match = re.search(r'<title>(.*?)</title>', text, re.DOTALL)
            date_match = re.search(r'<published_date>(.*?)</published_date>', text, re.DOTALL)
            categories_match = re.search(r'<categories>(.*?)</categories>', text, re.DOTALL)
            content_match = re.search(r'<content>(.*?)</content>', text, re.DOTALL)

            if not all([id_match, title_match, date_match, categories_match, content_match]):
                logging.error("Missing one or more required fields in tagged text.")
                return None

            id = int(id_match.group(1))
            title = title_match.group(1).strip()
            published_date = date_match.group(1).strip()
            categories_str = categories_match.group(1).strip()
            categories = re.findall(r'<(.*?)>', categories_str)
            content = content_match.group(1).strip()

            return cls(id, title, published_date, categories, content)

        except Exception as e:
            logging.error(f"Error parsing tagged text into Record: {e}")
            return None

    @classmethod
    def from_json(cls, json_str: str) -> Optional['Record']:
        """
        Initialize a Record object from a JSON string.
        
        :param json_str: JSON string representing the record.
        :return: Record object or None if parsing fails.
        """
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            logging.error(f"JSON decoding error: {e}")
            return None
        except Exception as e:
            logging.error(f"Error parsing JSON into Record: {e}")
            return None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional['Record']:
        """
        Initialize a Record object from a dictionary.
        
        :param data: Dictionary representing the record.
        :return: Record object or None if required fields are missing.
        """
        try:
            id = data['id']
            title = data['title']
            published_date = data['published_date']
            categories = data['categories']
            content = data['content']
            return cls(id, title, published_date, categories, content)
        except KeyError as e:
            logging.error(f"Missing required field in data dictionary: {e}")
            return None
        except Exception as e:
            logging.error(f"Error initializing Record from dict: {e}")
            return None

    @classmethod
    def from_unformatted_text(cls, text: str, llm_processor) -> Optional['Record']:
        """
        Initialize a Record object from unformatted text by using an LLM to structure it.
        
        :param text: Unformatted raw text.
        :param llm_processor: A callable that takes raw text and returns structured text.
        :return: Record object or None if processing fails.
        """
        try:
            # Use the LLM to format the unformatted text
            formatted_text = llm_processor(text)
            if not formatted_text:
                logging.error("LLM failed to format the unformatted text.")
                return None
            # Attempt to parse the formatted text as tagged text
            return cls.from_json(formatted_text)
        except Exception as e:
            logging.error(f"Error processing unformatted text into Record: {e}")
            return None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Record object to a dictionary.
        
        :return: Dictionary representation of the Record.
        """
        return {
            "id": self.id,
            "title": self.title,
            "published_date": self.published_date,
            "categories": self.categories,
            "content": self.content
        }

    def to_json(self) -> str:
        """
        Convert the Record object to a JSON string.
        
        :return: JSON string representation of the Record.
        """
        try:
            return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Error converting Record to JSON: {e}")
            return ""
        
    def get(self, key: Any, *args, default: Any = None) -> Any:
        """
        Retrieve a value from the Record similar to dict.get(), with support for list indices.

        :param key: The attribute name to retrieve or a list representing the path.
        :param args: Optional indices for list attributes.
        :param default: The value to return if the key/index is not found.
        :return: The retrieved value or default.
        """
        try:
            # If key is a list or tuple, treat it as a path
            if isinstance(key, (list, tuple)):
                keys = key
                if not keys:
                    logging.debug(f"Empty key path provided. Returning default: {default}")
                    return default
                first_key, *remaining_keys = keys
                value = getattr(self, first_key, default)
                if value is default:
                    logging.debug(f"Attribute '{first_key}' not found. Returning default: {default}")
                    return default
                for k in remaining_keys:
                    if isinstance(value, list):
                        if isinstance(k, int) and 0 <= k < len(value):
                            value = value[k]
                        else:
                            logging.debug(f"Index {k} out of range for attribute '{first_key}'. Returning default: {default}")
                            return default
                    elif isinstance(value, dict):
                        value = value.get(k, default)
                        if value is default:
                            logging.debug(f"Key '{k}' not found in dictionary. Returning default: {default}")
                            return default
                    else:
                        logging.debug(f"Attribute '{first_key}' is neither a list nor a dict. Cannot traverse key '{k}'. Returning default: {default}")
                        return default
                return value
            else:
                # Existing functionality
                value = getattr(self, key, default)
                if value is default:
                    logging.debug(f"Attribute '{key}' not found. Returning default: {default}")
                    return default

                for index in args:
                    if isinstance(value, list):
                        if isinstance(index, int) and 0 <= index < len(value):
                            value = value[index]
                        else:
                            logging.debug(f"Index {index} out of range for attribute '{key}'. Returning default: {default}")
                            return default
                    else:
                        logging.debug(f"Attribute '{key}' is not a list. Cannot apply index {index}. Returning default: {default}")
                        return default

                return value

        except Exception as e:
            record_id = self.id if hasattr(self, 'id') else 'N/A'
            logging.error(f"Error in get method for record ID {record_id}: {e}")
            return default