# src/utils/load_config.py
import yaml
import logging
import re
from dotenv import load_dotenv

import os
# logging.getLogger(__name__)

def load_config(config_path='src/config/config.yaml', dotenv_path='src/config/.env'):
    """
    Load the YAML configuration file and resolve environment variables.

    :param config_path: Path to the YAML configuration file.
    :param dotenv_path: Path to the .env file containing environment variables.
    :return: Configuration as a Python dictionary with environment variables substituted.
    """
    try:
        # Load environment variables from .env file if using python-dotenv
        if os.path.exists(dotenv_path):
            from dotenv import load_dotenv
            load_dotenv(dotenv_path)
            logging.info(f"Loaded environment variables from '{dotenv_path}'.")
        
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

        # Define a recursive function to substitute environment variables
        def substitute_env_vars(obj):
            if isinstance(obj, dict):
                return {k: substitute_env_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute_env_vars(element) for element in obj]
            elif isinstance(obj, str):
                # Find all occurrences of ${VAR_NAME}
                pattern = re.compile(r'\$\{([^}]+)\}')
                matches = pattern.findall(obj)
                for var in matches:
                    env_value = os.environ.get(var, "")
                    if not env_value:
                        logging.warning(f"Environment variable '{var}' is not set.")
                    obj = obj.replace(f"${{{var}}}", env_value)
                return obj
            else:
                return obj

        # Substitute environment variables in the config
        config = substitute_env_vars(config)

        # Validate that all required API keys are present
        required_providers = ['groq', 'google_gemini', 'ollama', 'openai']
        for provider in required_providers:
            if provider in config:
                api_key = config[provider].get('api_key', '')
                if not api_key:
                    logging.error(f"API key for provider '{provider}' is missing.")
                    raise ValueError(f"API key for provider '{provider}' is missing.")

        logging.info(f"Configuration loaded and environment variables substituted from '{config_path}'.")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file '{config_path}' not found.")
        raise
    except yaml.YAMLError as ye:
        logging.error(f"YAML parsing error in '{config_path}': {ye}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error loading configuration: {e}")
        raise
