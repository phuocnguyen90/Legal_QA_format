# utils/logging_setup.py

import logging
import sys

def setup_logging(log_file):
    """
    Configure logging to file and console.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Logging is set up.")
