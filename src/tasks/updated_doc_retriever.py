import os
import logging
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from difflib import SequenceMatcher, get_close_matches
import underthesea
from underthesea import word_tokenize  # Replace with appropriate tokenizer if needed
import pickle
from typing import List, Tuple, Dict
from fuzzywuzzy import fuzz


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def compute_tfidf_similarity(query, documents):
    vectorizer = TfidfVectorizer().fit(documents)
    query_vec = vectorizer.transform([query])
    doc_vecs = vectorizer.transform(documents)
    sims = cosine_similarity(query_vec, doc_vecs).flatten()
    return sims

def compute_similarity(query_text, document_text):
    # Use your preferred similarity measure, e.g., cosine similarity with TF-IDF
    vectorizer = TfidfVectorizer().fit([query_text, document_text])
    vectors = vectorizer.transform([query_text, document_text])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return similarity


class CombinedDocumentSearch:
    def __init__(self, csv_path: str, token_db_path: str = 'src/data/token_db.pkl'):
        """
        Initialize the CombinedDocumentSearch with paths to the CSV and token database.

        :param csv_path: Path to the CSV file containing documents.
        :param token_db_path: Path to the token database pickle file.
        """
        self.csv_path = csv_path
        self.token_db_path = token_db_path
        self.documents = self.load_documents()
        self.token_db = None
        self._load_or_create_token_database()

        self.equivalent_document_types = {
            'luật': ['luật', 'bộ luật', 'pháp lệnh'],
            'bộ luật': ['luật', 'bộ luật', 'pháp lệnh'],
            'pháp lệnh': ['luật', 'bộ luật', 'pháp lệnh'],
            # Add other mappings if necessary
        }

        # Initialize issuer mapping
        self.issuer_mapping = {
            'bộ công thương': 'bct',
            'bộ nội vụ': 'bnv',
            'bộ giáo dục': 'bgddt',
            'bộ tài chính': 'btc',
            'bộ quốc phòng': 'bqp',
            'bộ công an': 'bca',
            'bộ y tế': 'byt',
            'bộ thông tin': 'btttt',
            'bộ ngoại giao': 'bng',
            'bộ tư pháp': 'btp',
            'bộ kế hoạch': 'bkhdt',
            'bộ nông nghiệp': 'bnnptnt',
            'bộ giao thông': 'bgtvt',
            'bộ xây dựng': 'bxd',
            'bộ tài nguyên': 'btnmt',
            'bộ lao động': 'bldtbxh',
            'bộ văn hóa': 'bvhttdl',
            'bộ khoa học': 'bkhcn',
            'ngân hàng nhà nước': 'nhnn'

            # Add more mappings as necessary
        }

    def load_documents(self) -> pd.DataFrame:
        """
        Load documents from a CSV file into a DataFrame.

        :return: DataFrame of documents.
        """
        try:
            df = pd.read_csv(self.csv_path)
            logger.info(f"Loaded {len(df)} documents from {self.csv_path}.")

            # Normalize text columns: lowercase and strip whitespace
            text_columns = ['document_number', 'document_type', 'issuer_body', 'Full Name', 'Document_ID']
            for col in text_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.lower().str.strip()
                else:
                    logger.warning(f"Column '{col}' not found in DataFrame.")

            return df
        except Exception as e:
            logger.error(f"Error loading documents from {self.csv_path}: {e}")
            raise


    def _load_or_create_token_database(self):
        """
        Load or create the token database using TF-IDF.
        """
        if os.path.exists(self.token_db_path):
            logger.info("Loading token database from pickle file.")
            with open(self.token_db_path, 'rb') as f:
                self.token_db = pickle.load(f)
        else:
            logger.info("Creating new token database from document database.")
            documents = self.documents['Full Name'].tolist()
            self.token_db = self.create_token_database(documents, apply_tfidf=True)
            with open(self.token_db_path, 'wb') as f:
                pickle.dump(self.token_db, f)
            logger.info("Token database created and saved.")

    @staticmethod
    def create_token_database(documents, apply_tfidf=False):
        """
        Create a token frequency database from documents with optional TF-IDF weighting.
        
        :param documents: List of document titles.
        :param apply_tfidf: Whether to apply TF-IDF weighting.
        :return: Token frequency or TF-IDF scores dictionary.
        """
        tokenized_docs = [' '.join(word_tokenize(doc.lower())) for doc in documents]
        
        if apply_tfidf:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(tokenized_docs)
            feature_names = vectorizer.get_feature_names_out()
            
            tfidf_scores = {}
            for idx, token in enumerate(feature_names):
                tfidf_scores[token] = tfidf_matrix[:, idx].sum()
            return tfidf_scores
        else:
            token_freq = defaultdict(int)
            for doc in tokenized_docs:
                tokens = doc.split()
                for token in tokens:
                    token_freq[token] += 1
            return dict(token_freq)

    def search(self, query: str, top_n: int = 1, fuzzy: bool = True, cutoff: float = 0.5) -> Dict[str, List[Tuple[str, str, float]]]:
        matches = defaultdict(list)
        # Extract mentions along with their detailed information
        extracted_mentions = self.extract_document_mentions(query)

        for mention_dict in extracted_mentions:
            mention = mention_dict['mention']
            document_type = mention_dict['document_type']
            document_number = mention_dict['document_number']  # Add this line
            issue_year = mention_dict['issue_year']
            issuer_body = mention_dict['issuer_body']
            extra_info = mention_dict['extra_info']

            mention_matches = self.match_documents(
                mention, 
                document_type,
                document_number,     # Add this argument
                issue_year,
                issuer_body,
                extra_info,
                top_n=top_n,
                fuzzy=fuzzy,
                cutoff=cutoff
            )
            for key, value in mention_matches.items():
                matches[key].extend(value)

        # Sort and trim matches
        for mention, match_list in matches.items():
            sorted_match_list = sorted(match_list, key=lambda x: x[2], reverse=True)
            matches[mention] = sorted_match_list[:top_n]
            logger.info(f"Top {top_n} matches for mention '{mention}': {matches[mention]}")

        return matches



    def extract_document_mentions(self, text: str) -> List[Dict[str, str]]:
        mentions = []
        # Use sentence tokenizer
        sentences = underthesea.sent_tokenize(text)
        logger.debug(f"Sentences after tokenization: {sentences}")


        pattern = re.compile(
            r"(?P<document_type>Luật|Bộ luật|Pháp lệnh|Nghị định|Thông tư(?: liên tịch)?|Nghị quyết|Quyết định)"
            r"(?:\s+(?P<document_number>\d{1,4}))?"
            r"(?:\s+(?P<document_subtype>Luật|Bộ luật|Pháp lệnh))?"
            r"(?:\s+(?P<extra_info>.+?))?"
            r"(?:\s+(?P<issue_year>\d{4}))?",
            re.UNICODE | re.IGNORECASE
        )

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            # Search for all matches in the sentence
            for match in pattern.finditer(sentence):
                document_type = match.group("document_type").strip().lower() if match.group("document_type") else ""
                document_number = match.group("document_number").strip() if match.group("document_number") else ""
                document_subtype = match.group("document_subtype").strip().lower() if match.group("document_subtype") else ""
                issue_year = match.group("issue_year").strip() if match.group("issue_year") else ""
                extra_info = match.group("extra_info").strip() if match.group("extra_info") else ""

                # Build mention text
                mention_text = match.group(0).strip()

                # Determine the issuer body if present in extra info
                issuer_body = ""
                for body, abbrev in self.issuer_mapping.items():
                    if body in extra_info.lower():
                        issuer_body = abbrev
                        # Remove the identified issuer body from extra_info
                        extra_info = extra_info.replace(body, "").strip()

                # Adjust document type if document_subtype is present
                if document_subtype:
                    document_type += f" {document_subtype}"

                mentions.append({
                    'mention': mention_text,
                    'document_type': document_type,
                    'document_number': document_number,
                    'issue_year': issue_year,
                    'issuer_body': issuer_body,
                    'extra_info': extra_info
                })
                logger.debug(f"Extracted mention: {mentions[-1]}")

        logger.info(f"Extracted mentions with detailed info: {mentions}")
        return mentions


    def match_documents(
        self, 
        mention: str, 
        document_type: str,
        document_number: str,
        issue_year: str,
        issuer_body: str,
        extra_info: str,
        top_n: int = 2,
        fuzzy: bool = False,
        cutoff: float = 0.5  # Adjust as needed
    ) -> Dict[str, List[Tuple[str, str, float]]]:
        matches = defaultdict(list)

        # Assign base confidence scores
        base_confidences = {
            'document_type': 1.0 if document_type else 0.0,
            'document_number': 1.0 if document_number else 0.0,
            'issue_year': 1.0 if issue_year else 0.0,
            'issuer_body': 1.0 if issuer_body else 0.0,
            'extra_info': 1.0 if extra_info else 0.0
        }

        # Assign reliability factors
        reliability_factors = {
            'document_type': 0.8,
            'document_number': 0.9,
            'issue_year': 0.7,
            'issuer_body': 0.8,
            'extra_info': 0.6
        }

        # Calculate confidence scores
        confidence_scores = {}
        total_confidence = 0.0
        for prop in base_confidences:
            confidence = base_confidences[prop] * reliability_factors[prop]
            confidence_scores[prop] = confidence
            total_confidence += confidence

        # Normalize weights
        dynamic_weights = {}
        for prop in confidence_scores:
            if total_confidence > 0:
                dynamic_weights[prop] = confidence_scores[prop] / total_confidence
            else:
                dynamic_weights[prop] = 0.0

        # Build initial conditions based on available properties
        combined_condition = pd.Series(True, index=self.documents.index)
        for prop in ['document_type', 'document_number', 'issue_year', 'issuer_body']:
            value = locals()[prop]
            if value:
                if prop == 'document_type':
                    # Handle equivalent document types
                    equivalent_types = self.equivalent_document_types.get(value.lower(), [value.lower()])
                    condition = self.documents['document_type'].str.lower().isin(equivalent_types)
                else:
                    condition = self.documents[prop].str.lower() == value.lower()
                combined_condition &= condition

        # Apply conditions to get potential matches
        potential_matches = self.documents[combined_condition]
        logger.debug(f"Potential matches based on provided properties: {len(potential_matches)}")

        # Calculate match scores for potential matches
        for idx, doc in potential_matches.iterrows():
            match_score = 0.0

            # Check property matches
            for prop in ['document_type', 'document_number', 'issue_year', 'issuer_body']:
                value = locals()[prop]
                if value:
                    if prop == 'document_type':
                        match = doc['document_type'].lower() in self.equivalent_document_types.get(value.lower(), [value.lower()])
                    else:
                        match = doc[prop].lower() == value.lower()
                    if match:
                        match_score += dynamic_weights[prop]
                    else:
                        # Penalize mismatches
                        match_score -= dynamic_weights[prop]

            # Compute similarity for extra_info
            if extra_info:
                similarity = compute_similarity(extra_info.lower(), doc['Full Name'].lower())
                match_score += dynamic_weights['extra_info'] * similarity

            # Add match to results if score exceeds cutoff
            if match_score >= cutoff:
                matches[mention].append((doc['Full Name'], doc['Document_ID'], match_score))

        # Sort matches by score
        matches[mention] = sorted(matches[mention], key=lambda x: x[2], reverse=True)[:top_n]
        if matches[mention]:
            logger.info(f"Found matches for mention '{mention}' with dynamic scoring.")
        else:
            logger.info(f"No matches found for mention '{mention}'.")

        return matches




    def calculate_matching_score(self, doc: pd.Series, mention: str) -> float:
        """
        Calculate the matching score for a document based on title similarity, issue year, and other criteria.

        :param doc: The document row from the DataFrame.
        :param mention: The mention extracted from the query.
        :return: The calculated matching score.
        """
        # This method is now integrated into match_documents and can be removed or repurposed
        pass  # Placeholder if needed for future enhancements


    def extract_issue_year_from_mention(self, mention: str) -> str:
        """
        Extract the year from a document mention. For multiple mentions, match the found year
        with the nearest mention to the left of the year.

        :param mention: The document mention string.
        :return: The extracted year as a string, or None if not found.
        """
        year_pattern = r"/(\d{4})"
        matches = list(re.finditer(year_pattern, mention))
        if not matches:
            return None
        
        # If multiple years are found, match the year to the nearest mention on the left
        mention_positions = [m.start() for m in re.finditer(r"(?:Luật|Bộ luật|Pháp lệnh|Nghị định|Thông tư(?: liên tịch)?|Nghị quyết|Quyết định) \d{1,3}/\d{4}(?:/[\w\-]+)?", mention)]
        extracted_year = None
        for match in matches:
            year_pos = match.start()
            nearest_mention_pos = max((pos for pos in mention_positions if pos < year_pos), default=None)
            if nearest_mention_pos is not None:
                extracted_year = match.group(1)
                break
        
        return extracted_year

# Example usage
if __name__ == "__main__":
    searcher = CombinedDocumentSearch("src\data\doc_db_aggregated.csv")
    query = "Việc lập di chúc đều dựa trên mong muốn của chính người lập di chúc, tuy nhiên, vẫn tồn tại những trường hợp thay đổi hoặc huỷ bỏ di chúc đã được xác lập vì nhiều lí do khác nhau và dựa trên ý chỉ của chủ thể.\n\nNgười lập di chúc có thể sửa đổi, bổ sung di chúc bất kỳ lúc nào? Theo khoản 1 Điều 640 Bộ luật dân sự 2015, người lập di chúc có thể sửa đổi, bổ sung, thay thế, huỷ bỏ di chúc đã lập vào bất cứ lúc nào.\n\nNếu một người lập nhiều di chúc thì bản di chúc nào sẽ có hiệu lực pháp luật? Căn cứ tại khoản 5 Điều 643 Bộ luật Dân sự 2015 quy định về hiệu lực của di chúc thì khi một người để lại nhiều bản di chúc đối với một tài sản thì chỉ bản di chúc sau cùng có hiệu lực.\n\nCông chứng di chúc được sửa đổi. Căn cứ theo khoản 3 Điều 56 Luật Công chứng 2014 quy định về công chứng di chúc như sau: “Điều 56. Công chứng di chúc...”\n\nHồ sơ cần chuẩn bị cho việc thay đổi hoặc huỷ bỏ di chúc. Dự thảo di chúc hoặc bản di chúc đã lập trước đó (di chúc cần sửa đổi, bổ sung). Phiếu yêu cầu công chứng trong đó nêu rõ yêu cầu lập di chúc mới hay sửa đổi, bổ sung di chúc đã lập. Giấy tờ cá nhân của người lập/sửa đổi di chúc và người được hưởng thừa kế theo di chúc.."
    results = searcher.search(query, fuzzy=False)
    print(results)