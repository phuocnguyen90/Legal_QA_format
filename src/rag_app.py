import os
import logging
import fastembed
import numpy as np
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from typing import Dict, Any
from groq import Groq  # Ensure Groq SDK is installed

from utils.load_config import load_config

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration from config.yaml
try:
    config = load_config('src/config/config.yaml')
    logger.info("Configuration loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    exit(1)

# Load environment variables
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

# Verify that required environment variables are provided
if not QDRANT_API_KEY or not QDRANT_URL:
    logger.error("QDRANT_API_KEY and QDRANT_URL must be set.")
    exit(1)

# Qdrant Collection Configuration
COLLECTION_NAME = "presidential_speeches"
DIMENSION = 384  # Updated to match the FastEmbed model's output dimension

# Set up Qdrant Client with Qdrant Cloud parameters
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=True  # Optional, faster if using gRPC protocol, but make sure Qdrant Cloud supports it
)

# Ensure the collection exists, otherwise create it
try:
    if qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
        # Optionally delete the collection if you need to recreate it
        # qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
        logger.info(f"Qdrant collection '{COLLECTION_NAME}' already exists.")
    else:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qdrant_models.VectorParams(size=DIMENSION, distance="Cosine")
        )
        logger.info(f"Qdrant collection '{COLLECTION_NAME}' created with dimension {DIMENSION}.")
except Exception as e:
    logger.error(f"Failed to create or verify Qdrant collection: {e}")
    exit(1)

# Initialize the Groq Client for LLM API calls
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
    model = config.get('groq_model_name', 'llama3-8b-8192')
    logger.info("Groq client initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {e}")
    exit(1)

# Step 1: Embedding Function using FastEmbed
def fe_embed_text(text: str) -> list:
    """
    Embed a single text using FastEmbed with a small multilingual model.

    :param text: A single text string to embed.
    :return: A list of floats representing the embedding vector.
    """
    try:
        # Initialize the FastEmbed embedding model once
        if not hasattr(fe_embed_text, "embedding_model"):
            fe_embed_text.embedding_model = fastembed.TextEmbedding(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        
        # Obtain the embedding generator
        embedding_generator = fe_embed_text.embedding_model.embed(text)
        
        # Convert the generator to a list and then to a numpy array
        embeddings = list(embedding_generator)
        if not embeddings:
            logger.error(f"No embeddings returned for text '{text}'.")
            return []
        
        embedding = np.array(embeddings)
        
        # Handle 2D arrays with a single embedding vector
        if embedding.ndim == 2 and embedding.shape[0] == 1:
            embedding = embedding[0]
        
        # Ensure that the embedding is a flat array
        if embedding.ndim != 1:
            logger.error(f"Embedding for text '{text}' is not a flat array. Got shape: {embedding.shape}")
            return []
        
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Failed to create embedding for the input: '{text}', error: {e}")
        return []

# Step 2: Function to Add a Single Document to Qdrant
def add_document_to_qdrant(document: Dict[str, Any]):
    """
    Add a single document to the Qdrant collection.

    :param document: A dictionary containing 'content' and 'metadata'.
    """
    content = document['content']
    metadata = document.get('metadata', {})

    embedding = fe_embed_text(content)
    if not embedding:
        logger.error("Skipping document due to embedding failure.")
        return

    # Generate a unique UUID for each document
    point_id = str(uuid.uuid4())

    # Include 'content' in the payload alongside 'metadata'
    payload = {
        "content": content,
        **metadata  # Unpack metadata into the payload
    }

    point = qdrant_models.PointStruct(
        id=point_id,
        vector=embedding,
        payload=payload
    )

    # Insert data into Qdrant
    try:
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point]
        )
        logger.info(f"Successfully added document with ID {point_id} to Qdrant collection '{COLLECTION_NAME}'.")
    except Exception as e:
        logger.error(f"Failed to add document to Qdrant: {e}")


# Step 3: Function to Perform Vector Search in Qdrant
def search_qdrant(query: str, top_k: int = 3) -> list:
    """
    Search Qdrant using an embedding of the query.

    :param query: The input query to search similar documents.
    :param top_k: Number of similar results to return.
    :return: List of similar documents with 'content' and 'source'.
    """
    query_embedding = fe_embed_text(query)

    if not query_embedding:
        logger.error("Failed to create a valid embedding for query.")
        return []

    try:
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k
        )
        # Extract 'content' and 'source' from the payload
        return [
            {
                "content": hit.payload.get("content", ""),
                "source": hit.payload.get("source", ""),
                "score": hit.score
            }
            for hit in search_result
        ]
    except Exception as e:
        logger.error(f"Error searching Qdrant: {e}")
        return []



def rag(query: str) -> str:
    """
    Use Qdrant for retrieval and Groq LLM for augmented generation.

    :param query: The query to generate a response for.
    :return: Augmented response from LLM.
    """
    # Step 4.1: Retrieve similar documents using Qdrant
    retrieved_docs = search_qdrant(query)
    if not retrieved_docs:
        return "No relevant information found."

    # Step 4.2: Combine retrieved documents to form context for LLM
    context = "\n\n------------------------------------------------------\n\n".join(
        [f"Source: {doc['source']}\nContent: {doc['content']}" for doc in retrieved_docs if doc['content']]
    )

    # Step 4.3: Query the Groq LLM with retrieved context and original query
    system_prompt = '''
    You are a presidential historian. Given the user's question and relevant excerpts from 
    presidential speeches, answer the question by including direct quotes from presidential speeches. 
    When using a quote, cite the speech that it was from.
    '''

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"User Question: {query}\n\nRelevant Speech Excerpt(s):\n\n{context}",
                }
            ],
            model=model
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating response from Groq LLM: {e}")
        return "An error occurred while generating the response."


# Step 5: Example Usage
if __name__ == "__main__":
    # Example documents to add to Qdrant
    documents = [
        {"content": "George Washington emphasized the importance of democracy in his farewell address.", "metadata": {"source": "Washington Farewell Speech"}},
        {"content": "Abraham Lincoln spoke about national unity in his Gettysburg Address.", "metadata": {"source": "Lincoln Gettysburg Address"}},
        {"content": "Thomas Jefferson's inaugural address mentioned the importance of equality.", "metadata": {"source": "Jefferson Inaugural Address"}},
    ]

    # Add documents to Qdrant one by one
    for document in documents:
        add_document_to_qdrant(document)

    # Test the RAG function
    user_query = "What did Abraham Lincoln say about national unity?"
    answer = rag(user_query)
    print(f"Answer: {answer}")
