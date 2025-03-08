import logging
import os
import sys
import json
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

# Constants
API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "datagram-index"
MAX_BATCH_SIZE = 4 * 1024 * 1024  # 4MB batch size limit
EMBEDDING_DIMENSION = 1024
FETCH_BATCH_SIZE = 10000  # Number of IDs to fetch per batch

if not API_KEY:
    raise EnvironmentError("Pinecone API key is not set.")

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Pinecone client
pc = Pinecone(api_key=API_KEY)

# Access Pinecone index
try:
    index = pc.Index(INDEX_NAME)
    logging.info(f"Pinecone index '{INDEX_NAME}' accessed successfully.")
except Exception as e:
    logging.error(f"Error accessing Pinecone index '{INDEX_NAME}': {e}")
    raise

def get_all_vector_ids(namespace):
    """
    Retrieve all vector IDs from a specific namespace using fetch operation.
    """
    try:
        all_ids = []
        cursor = None
        
        while True:
            # Fetch batch of vectors
            fetch_response = index.fetch(
                ids=[],  # Empty list to fetch all vectors
                namespace=namespace,
                limit=FETCH_BATCH_SIZE,
                cursor=cursor
            )
            
            # Extract vector IDs from the response
            batch_ids = list(fetch_response.vectors.keys())
            all_ids.extend(batch_ids)
            
            # Update cursor for next batch
            cursor = fetch_response.cursor
            logging.info(f"Fetched {len(batch_ids)} vector IDs, total: {len(all_ids)}")
            
            # Break if no more vectors
            if not cursor or not batch_ids:
                break
                
        logging.info(f"Retrieved total of {len(all_ids)} vector IDs from namespace '{namespace}'")
        return all_ids
    except Exception as e:
        logging.error(f"Error fetching vector IDs: {str(e)}", exc_info=True)
        return []

def split_list(lst, n):
    """Split a list into n-sized chunks."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def queryIndex(query_embedding, namespace):
    """
    Perform comprehensive semantic search on the Pinecone index using query embedding.
    Retrieves and scores all vectors in the namespace.
    
    Args:
        query_embedding: The embedding vector to search with
        namespace: The namespace to search in
    """
    try:
        # First, get all vector IDs in the namespace
        all_vector_ids = get_all_vector_ids(namespace)
        if not all_vector_ids:
            return {
                "status": "success",
                "results": [],
                "total_vectors": 0
            }
        
        # Split vector IDs into manageable batches (Pinecone has a limit on number of IDs per query)
        id_batches = list(split_list(all_vector_ids, 1000))
        all_matches = []
        
        # Query each batch of IDs
        for batch_num, id_batch in enumerate(id_batches, 1):
            results = index.query(
                vector=query_embedding,
                namespace=namespace,
                id=id_batch,  # Explicitly specify which vectors to score
                include_metadata=True,
                top_k=len(id_batch),  # Request scores for all IDs in batch
                filter={},  # Empty filter to ensure no filtering
            )
            
            # Add debug logging
            logging.debug(f"Batch {batch_num} query response: {results}")
            
            if 'matches' in results:
                batch_matches = results['matches']
                all_matches.extend(batch_matches)
                logging.info(f"Processed batch {batch_num}/{len(id_batches)}, "
                           f"got {len(batch_matches)} matches. "
                           f"Running total: {len(all_matches)}")
            else:
                logging.warning(f"No 'matches' in response for batch {batch_num}: {results}")

        # Sort all matches by score in descending order
        all_matches.sort(key=lambda x: x['score'], reverse=True)
        
        logging.info(f"Query successful. Retrieved and scored {len(all_matches)} vectors")
        
        # Add detailed response information
        return {
            "status": "success",
            "results": all_matches,
            "total_vectors": len(all_matches),
            "total_batches_processed": len(id_batches),
            "vectors_per_batch": [len(batch) for batch in id_batches]
        }
        
    except Exception as e:
        logging.error(f"Error performing semantic search: {str(e)}", exc_info=True)
        return {
            "status": "fail",
            "message": f"Error performing semantic search: {str(e)}"
        }

def store_embeddings(embeddings, namespace):
    """
    Store embeddings into the specified Pinecone index with namespace for identification.
    """
    try:
        vectors = []
        for i, embedding in enumerate(embeddings):
            vectors.append({
                "id": embedding.get('id', f"{namespace}_vec_{i}"),
                "values": embedding['values'],
                "metadata": embedding.get('metadata', {})
            })

        batches = calculate_batch_size(vectors)
        total_upserted = 0
        for batch in batches:
            index.upsert(vectors=batch, namespace=namespace)
            total_upserted += len(batch)

        logging.info(f"Successfully stored {total_upserted} embeddings under namespace '{namespace}'.")
        return {"status": "success", "num_embeddings": total_upserted}
    except Exception as e:
        logging.error(f"Error storing embeddings: {str(e)}", exc_info=True)
        return {"status": "fail", "message": f"Error storing embeddings: {str(e)}"}

def calculate_batch_size(data, max_batch_size=MAX_BATCH_SIZE):
    """
    Splits data into batches under the specified size limit (default: 4MB).
    """
    batches = []
    current_batch = []
    current_size = 0

    for item in data:
        item_size = sys.getsizeof(json.dumps(item))

        if current_size + item_size > max_batch_size:
            batches.append(current_batch)
            current_batch = [item]
            current_size = item_size
        else:
            current_batch.append(item)
            current_size += item_size

    if current_batch:
        batches.append(current_batch)

    return batches