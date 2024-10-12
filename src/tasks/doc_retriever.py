import logging
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import re
import pickle
from underthesea import word_tokenize  # Replace with appropriate tokenizer if needed
from difflib import get_close_matches
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Class for Document Retrieval
class DocRetriever:
    def __init__(self, config):
        self.config=config
        self.document_db_path = self.config['processing']['document_db']
        self.token_db_path = self.config['processing'].get('token_db', 'data/token_db.pkl')
        self.df = None
        self.token_db = None
        self._load_or_create_token_database()

    def _load_or_create_token_database(self):
        """
        Load an existing token database from a pickle file, or create a new one if not present.
        """
        if os.path.exists(self.token_db_path):
            logging.debug("Loading token database from pickle file.")
            with open(self.token_db_path, 'rb') as f:
                self.token_db = pickle.load(f)
        else:
            logging.debug("Creating new token database from document database.")
            self.df = pd.read_csv(self.document_db_path)
            documents = self.df['Full Name'].tolist()
            self.token_db = self.create_token_database(documents, apply_tfidf=True)
            with open(self.token_db_path, 'wb') as f:
                pickle.dump(self.token_db, f)
            logging.debug("Token database created and saved.")

    @staticmethod
    def create_token_database(documents, apply_tfidf=False):
        """
        Create a token frequency database from a list of documents, with optional TF-IDF weighting.
        
        Args:
            documents (list of str): List of text documents.
            apply_tfidf (bool): Whether to apply TF-IDF weighting to the tokens.
            
        Returns:
            dict: A dictionary where keys are tokens and values are their frequency or TF-IDF score.
        """
        logging.debug("Starting tokenization of documents.")
        tokenized_docs = [' '.join(word_tokenize(doc.lower())) for doc in documents]
        
        if apply_tfidf:
            logging.debug("Applying TF-IDF weighting.")
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(tokenized_docs)
            feature_names = vectorizer.get_feature_names_out()
            
            tfidf_scores = {}
            for idx, token in enumerate(feature_names):
                tfidf_scores[token] = tfidf_matrix[:, idx].sum()
            logging.debug("TF-IDF token database created.")
            return tfidf_scores
        else:
            logging.debug("Creating token frequency dictionary.")
            token_freq = defaultdict(int)
            for doc in tokenized_docs:
                tokens = doc.split()
                for token in tokens:
                    token_freq[token] += 1
            logging.debug("Token frequency dictionary created.")
            return dict(token_freq)

    def search(self, query, top_n=5, fuzzy=False, cutoff=0.8, document_type=None):
        """
        Search for documents based on the given query.
        
        Args:
            query (str): The user query for matching.
            top_n (int): Number of top matches to return per mention.
            fuzzy (bool): Whether to use fuzzy matching.
            cutoff (float): Threshold for fuzzy matching.
            document_type (str or None): Specify the type of document to filter results.
            
        Returns:
            dict: A mapping of matched mentions to lists of (Full Name, Document_ID) tuples.
        """
        logging.debug("Starting document search.")
        if self.df is None:
            self.df = pd.read_csv(self.document_db_path)
        
        return self.match_documents_comprehensive(query, self.token_db, {}, self.df, top_n, fuzzy, cutoff, document_type)

    @staticmethod
    def match_documents_comprehensive(text, inverted_index, document_types, df, top_n=5, fuzzy=False, cutoff=0.8, document_type=None):
        """
        Comprehensive matching function combining rule-based and NLP approaches.
        
        Args:
            text (str): The input text containing partial mentions.
            inverted_index (dict): The inverted index mapping trigrams to document IDs.
            document_types (dict): The mapping of document types to document IDs.
            df (pd.DataFrame): The DataFrame containing document data.
            top_n (int): Number of top matches to return per mention.
            fuzzy (bool): Whether to use fuzzy matching.
            cutoff (float): Threshold for fuzzy matching.
            document_type (str or None): Specify the type of document to filter results.
            
        Returns:
            dict: A mapping of matched mentions to lists of (Full Name, Document_ID) tuples.
        """
        logging.debug("Initializing matches dictionary.")
        matches = defaultdict(list)
        normalized_text = normalize_text(text)
        
        logging.debug("Extracting mentions using regex.")
        partial_pattern = re.compile(
            r"(Luật|Bộ luật|Pháp lệnh|Nghị định|Thông tư(?: liên tịch)?|Nghị quyết|Quyết định)\s+\d{1,3}/\d{4}(?:/[\w\-]+)?",
            re.UNICODE | re.IGNORECASE
        )
        
        regex_matches = partial_pattern.finditer(normalized_text)
        mentions = set()
        for match in regex_matches:
            full_mention = match.group(0)
            mentions.add(full_mention)
        logging.debug(f"Extracted mentions: {mentions}")
        
        logging.debug("Tokenizing text and extracting trigrams.")
        tokens = word_tokenize(normalized_text)
        trigrams = extract_trigrams(tokens)
        logging.debug(f"Extracted trigrams: {trigrams}")
        
        logging.debug("Retrieving candidate documents based on trigrams.")
        candidate_docs = defaultdict(int)
        for trigram in trigrams:
            if trigram in inverted_index:
                for doc_id in inverted_index[trigram]:
                    candidate_docs[doc_id] += 1
        logging.debug(f"Candidate documents: {candidate_docs}")
        
        logging.debug("Ranking candidates based on trigram matches.")
        sorted_candidates = sorted(candidate_docs.items(), key=lambda x: x[1], reverse=True)
        logging.debug(f"Sorted candidates: {sorted_candidates}")
        
        logging.debug("Applying matching rules for each mention.")
        first_category_keywords = {"luật", "bộ luật", "pháp lệnh"}
        for mention in mentions:
            logging.debug(f"Processing mention: {mention}")
            m = partial_pattern.match(mention)
            if m:
                doc_type = m.group(1).lower()
                id_pattern = re.compile(r"\d{1,3}/\d{4}(?:/[A-Za-z\-]+)?")
                id_match = id_pattern.search(mention)
                partial_id = id_match.group(0).lower() if id_match else ""
                logging.debug(f"Extracted doc_type: {doc_type}, partial_id: {partial_id}")
                if document_type and document_type.lower() != doc_type:
                    continue
                if doc_type in first_category_keywords:
                    logging.debug("Matching first category document.")
                    exact_matches = df[df['Document_ID'].str.lower() == partial_id]
                    if not exact_matches.empty:
                        for _, row in exact_matches.iterrows():
                            matches[mention].append((row['Full Name'], row['Document_ID']))
                        logging.debug(f"Exact matches found: {matches[mention]}")
                        continue
                    if fuzzy and partial_id:
                        logging.debug("Performing fuzzy match for partial ID.")
                        possible_ids = get_close_matches(partial_id, df['Document_ID'].str.lower(), n=top_n, cutoff=cutoff)
                        for pid in possible_ids:
                            matched_doc = df[df['Document_ID'].str.lower() == pid]
                            if not matched_doc.empty:
                                matches[mention].append((matched_doc.iloc[0]['Full Name'], matched_doc.iloc[0]['Document_ID']))
                        logging.debug(f"Fuzzy matches found: {matches[mention]}")
                        if possible_ids:
                            continue
                    if partial_id:
                        logging.debug("Performing partial ID match using regex.")
                        partial_id_pattern = re.compile(re.escape(partial_id) + r'.*')
                        partial_matches = df[df['Document_ID'].str.lower().str.match(partial_id_pattern)]
                        if not partial_matches.empty:
                            for _, row in partial_matches.iterrows():
                                matches[mention].append((row['Full Name'], row['Document_ID']))
                            logging.debug(f"Partial matches found: {matches[mention]}")
                            continue
                else:
                    logging.debug("Matching second category document.")
                    exact_matches = df[df['Document_ID'].str.lower() == partial_id]
                    if not exact_matches.empty:
                        for _, row in exact_matches.iterrows():
                            matches[mention].append((row['Full Name'], row['Document_ID']))
                        logging.debug(f"Exact matches found: {matches[mention]}")
                        continue
                    if fuzzy and partial_id:
                        logging.debug("Performing fuzzy match for partial ID.")
                        possible_ids = get_close_matches(partial_id, df['Document_ID'].str.lower(), n=top_n, cutoff=cutoff)
                        for pid in possible_ids:
                            matched_doc = df[df['Document_ID'].str.lower() == pid]
                            if not matched_doc.empty:
                                matches[mention].append((matched_doc.iloc[0]['Full Name'], matched_doc.iloc[0]['Document_ID']))
                        logging.debug(f"Fuzzy matches found: {matches[mention]}")
                        if possible_ids:
                            continue
                    if partial_id:
                        logging.debug("Performing partial ID match using regex.")
                        partial_id_pattern = re.compile(re.escape(partial_id) + r'.*')
                        partial_matches = df[df['Document_ID'].str.lower().str.match(partial_id_pattern)]
                        if not partial_matches.empty:
                            for _, row in partial_matches.iterrows():
                                matches[mention].append((row['Full Name'], row['Document_ID']))
                            logging.debug(f"Partial matches found: {matches[mention]}")
                            continue
        if not any(matches.values()):
            logging.debug("Checking for trigram-based candidates.")
            for doc_id, count in sorted_candidates[:top_n]:
                matched_doc = df[df['Document_ID'].str.lower() == doc_id.lower()]
                if not matched_doc.empty:
                    mention = f"Candidate Match for Trigram: {doc_id}"
                    matches[mention].append((matched_doc.iloc[0]['Full Name'], matched_doc.iloc[0]['Document_ID']))
            logging.debug(f"Trigram-based matches found: {matches}")
        return matches

# Utility functions
def normalize_text(text):
    """
    Normalize the text by lowercasing and removing extra spaces.
    """
    return ' '.join(text.lower().split())

def extract_trigrams(tokens):
    """
    Extract trigrams from a list of tokens.
    """
    trigrams = []
    for i in range(len(tokens) - 2):
        trigram = ' '.join(tokens[i:i+3])
        trigrams.append(trigram)
    return trigrams