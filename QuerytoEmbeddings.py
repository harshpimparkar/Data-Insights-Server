import os
import logging
from mixedbread_ai.client import MixedbreadAI

# Initialize MixedbreadAI client
MXB_API_KEY = os.getenv("MIXEDBREAD_API_KEY")
if not MXB_API_KEY:
    raise EnvironmentError("MIXEDBREAD_API_KEY is not set in the environment variables.")
mxbai = MixedbreadAI(api_key=MXB_API_KEY)

def query_to_embedding(query):
    """
    Converts a query string into an embedding using MixedbreadAI.
    """
    try:
        # Generate embedding
        res = mxbai.embeddings(
            model="mixedbread-ai/mxbai-embed-large-v1",
            input=[query],
            normalized=True,
            encoding_format="float",
            truncation_strategy="end"
        )
        
        if res.data:
            embedding = res.data[0].embedding
            logging.info("Embedding successfully generated: %s, query: %s", embedding, query)
            return {"status": "success", "embedding": embedding}
        else:
            logging.warning("No embedding generated for the query: %s", query)
            return {"status": "fail", "message": "No embedding generated for the query."}
    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        return {"status": "fail", "message": str(e)}
