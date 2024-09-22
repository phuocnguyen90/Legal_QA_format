import os
import re
import logging
from groq import Groq
from tqdm import tqdm
import time

import getpass
# Configure logging
logging.basicConfig(
    filename='/content/drive/My Drive/Colab_Preprocessed_Files/preprocessing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
GROQ_API_KEY = getpass.getpass('Enter your Groq API Key: ')

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

def read_cleaned_data(file_path):
    """
    Reads the entire content of the cleaned text file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.read()
        logging.info(f"Successfully read cleaned data from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error reading cleaned file {file_path}: {e}")
        raise

def split_records(data):
    """
    Splits the data into individual records using the <id=number></id=number> tags.
    """
    # Regular expression pattern to match each record
    pattern = r'(<id=\d+>.*?</id=\d+>)'
    
    # Find all matches with re.DOTALL to include newlines
    records = re.findall(pattern, data, re.DOTALL)
    
    logging.info(f"Total records found: {len(records)}")
    return records

def detect_language(text):
    """
    Detects the language of the given text.
    """
    try:
        return detect(text)
    except Exception as e:
        logging.warning(f"Language detection failed: {e}")
        return "unknown"
def translate_to_vietnamese(text, client):
    """
    Translates English text to Vietnamese using the Groq API.
    """
    prompt = f"""Translate the following English text to Vietnamese. Ensure that the translation is accurate and maintains the original meaning without any factual inconsistencies.

Original Text:
\"\"\"
{text}
\"\"\"
Translated Text:"""
    
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,  # Adjust based on the size of your data and model limits
            temperature=0.3
        )
        translated_text = response.choices[0].message.content.strip()
        return translated_text
    except Exception as e:
        logging.error(f"Error translating text: {e}")
        return text  # Return original text if translation fails

def fact_check(original_text, translated_text, client):
    """
    Performs fact checking to ensure factual consistency between original and translated text.
    """
    prompt = f"""Perform a factual consistency check between the following two texts. Identify any factual inconsistencies or discrepancies.

Original Text:
\"\"\"
{original_text}
\"\"\"

Translated Text:
\"\"\"
{translated_text}
\"\"\"

Please respond with 'Consistent' if there are no factual inconsistencies, or provide details of any inconsistencies found."""
    
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.0  # Lower temperature for factual accuracy
        )
        result = response.choices[0].message.content.strip()
        if "Consistent" in result:
            return True
        else:
            logging.warning("Factual inconsistencies detected.")
            return False
    except Exception as e:
        logging.error(f"Error during fact checking: {e}")
        return False

def process_record_post(cleaned_record, client):
    """
    Processes a single cleaned record: detects language, translates if necessary, and fact checks.
    """
    # Extract content for language detection
    content_match = re.search(r'<content>(.*?)</content>', cleaned_record, re.DOTALL)
    if not content_match:
        logging.warning("No <content> tag found. Skipping language detection and translation.")
        return cleaned_record  # Return cleaned record as is
    
    content = content_match.group(1).strip()
    
    # Detect language
    language = detect_language(content)
    logging.info(f"Detected language: {language}")
    
    if language == 'en':
        logging.info("English content detected. Proceeding to translate to Vietnamese.")
        # Translate to Vietnamese
        translated_content = translate_to_vietnamese(content, client)
        
        # Fact Check
        is_consistent = fact_check(content, translated_content, client)
        if not is_consistent:
            # Append a note for manual review
            translated_content += "\n\n<!-- Factual inconsistencies detected. Please review. -->"
        
        # Replace original content with translated content
        updated_record = re.sub(r'(<content>)(.*?)(</content>)', r'\1' + translated_content + r'\3', cleaned_record, flags=re.DOTALL)
        return updated_record
    elif language == 'vi':
        logging.info("Content is already in Vietnamese. No translation needed.")
        return cleaned_record
    else:
        logging.warning(f"Unknown language detected: {language}. Skipping translation.")
        return cleaned_record  # Return cleaned record as is
def save_final_record(processed_record, final_output_file):
    """
    Appends the final processed record to the final output file.
    """
    try:
        with open(final_output_file, 'a', encoding='utf-8') as f:
            f.write(processed_record + '\n\n')
        logging.info("Final processed record saved successfully.")
    except Exception as e:
        logging.error(f"Error saving final processed record: {e}")

def main_postprocessing():
    # Define input and output file paths
    input_file = '/content/drive/My Drive/Colab_Preprocessed_Files/cleaned_result.txt'  # Path to cleaned data
    output_file = '/content/drive/My Drive/Colab_Postprocessed_Files/final_result.txt'
    
    # Ensure the output file is empty before starting
    open(output_file, 'w', encoding='utf-8').close()
    logging.info(f"Final output file {output_file} initialized.")
    
    # Step 1: Read cleaned data from the input file
    cleaned_data = read_cleaned_data(input_file)
    
    # Step 2: Split cleaned data into records
    records = split_records(cleaned_data)
    
    # Verify if records were split correctly
    if not records:
        logging.error("No records found in cleaned data. Please check the input file format.")
        print("No records found in cleaned data. Please check the input file format.")
        return
    
    # Step 3: Process each record
    total_records = len(records)
    
    for idx, record in enumerate(tqdm(records, desc="Post-processing Records"), start=1):
        logging.info(f"Post-processing record {idx}/{total_records}")
        print(f"Post-processing record {idx}/{total_records}")
        
        processed_record = process_record_post(record, client)
        
        if processed_record:
            save_final_record(processed_record, output_file)
        else:
            logging.warning(f"Record {idx} was not post-processed successfully.")
            print(f"Record {idx} was not post-processed successfully.")
        
        # Optional: Sleep to respect API rate limits
        time.sleep(3)  # Adjust as needed based on API rate limits
    
    print(f"Post-processing complete. Final data saved to {output_file}")
    logging.info("All records post-processed successfully.")
# Execute the main post-processing function
main_postprocessing()
