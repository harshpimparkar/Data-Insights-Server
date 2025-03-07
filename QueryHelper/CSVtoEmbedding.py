import os
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from mixedbread_ai.client import MixedbreadAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model: str = 'mixedbread-ai/mxbai-embed-large-v1'
    chunk_size: int = 100  # Reduced chunk size for better reliability
    max_text_length: int = 8192
    max_workers: int = min(os.cpu_count() or 4, 8)
    retry_attempts: int = 3

class EmbeddingGenerator:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the embedding generator with API key and configuration."""
        self.api_key = api_key or os.getenv("MIXEDBREAD_API_KEY")
        if not self.api_key:
            raise EnvironmentError("MIXEDBREAD_API_KEY is not set")
        
        self.client = MixedbreadAI(api_key=self.api_key)
        self.config = EmbeddingConfig()
        
    def _clean_text(self, text: Any) -> str:
        """Clean and validate text input."""
        if pd.isna(text):
            return ""
        
        text = str(text).strip()
        # Truncate if exceeds maximum length
        if len(text) > self.config.max_text_length:
            logging.warning(f"Text truncated from {len(text)} to {self.config.max_text_length} characters")
            text = text[:self.config.max_text_length]
            
        return text

    def _create_text_batch(self, chunk: pd.DataFrame) -> List[str]:
        """Create cleaned text representations for embedding."""
        text_entries = []
        
        for _, row in chunk.iterrows():
            entry_parts = []
            for col, val in row.items():
                cleaned_val = self._clean_text(val)
                if cleaned_val:
                    entry_parts.append(f"{col}: {cleaned_val}")
            
            if entry_parts:
                text_entries.append(" | ".join(entry_parts))
        
        return text_entries

    @retry(
        stop=stop_after_attempt(EmbeddingConfig.retry_attempts),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def _get_embeddings(self, text_data: List[str]) -> Any:
        """Get embeddings with retry logic and error handling."""
        try:
            response = self.client.embeddings(
                model=self.config.model,
                input=text_data,
                normalized=True,
                encoding_format='float',
                truncation_strategy='end'
            )
            
            if not response.data:
                raise ValueError("API returned empty embedding data")
                
            return response
            
        except Exception as e:
            logging.error(f"Embedding API error: {str(e)}")
            if hasattr(e, 'response'):
                logging.error(f"Response details: {e.response.text}")
            raise

    def _process_chunk(self, chunk: pd.DataFrame, start_idx: int) -> List[Dict[str, Any]]:
        """Process a data chunk with comprehensive error handling."""
        if chunk.empty:
            logging.warning(f"Empty chunk received at index {start_idx}")
            return []

        try:
            # Create and validate text batch
            text_data = self._create_text_batch(chunk)
            if not text_data:
                logging.warning(f"No valid text created from chunk at index {start_idx}")
                return []

            # Process in smaller batches for reliability
            results = []
            for i in range(0, len(text_data), self.config.chunk_size):
                batch = text_data[i:i + self.config.chunk_size]
                try:
                    response = self._get_embeddings(batch)
                    
                    # Process successful embeddings
                    batch_results = [
                        {
                            "id": str(start_idx + i + idx),
                            "values": item.embedding,
                            "metadata": {
                                **{
                                    col: float(val) if isinstance(val, (float, int)) else str(val)
                                    for col, val in row.items()
                                    if pd.notna(val)
                                },
                                "full_text": text
                            }
                        }
                        for idx, (_, row), item, text in zip(
                            itertools.count(),
                            chunk.iloc[i:i + self.config.chunk_size].iterrows(),
                            response.data,
                            batch
                        )
                        if item.embedding is not None
                    ]
                    
                    results.extend(batch_results)
                    logging.info(f"Processed batch of {len(batch_results)} embeddings at index {start_idx + i}")
                    
                except RetryError as e:
                    logging.error(f"Failed to process batch after {self.config.retry_attempts} attempts at index {start_idx + i}")
                    continue
                except Exception as e:
                    logging.error(f"Error processing batch at index {start_idx + i}: {str(e)}")
                    continue

            return results

        except Exception as e:
            logging.error(f"Error in chunk processing at index {start_idx}: {str(e)}")
            return []

    def generate_embeddings(self, file_path: str) -> Dict[str, Any]:
        """Generate embeddings from CSV file with improved error handling and reporting."""
        try:
            # Validate file
            if not os.path.exists(file_path):
                return {"status": "error", "message": f"File not found: {file_path}"}
            
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return {"status": "error", "message": "File is empty"}

            # Get file metadata
            total_rows = sum(1 for _ in open(file_path)) - 1
            if total_rows <= 0:
                return {"status": "error", "message": "File contains no data rows"}

            # Initialize processing
            logging.info(f"Processing file: {file_path}")
            logging.info(f"Total rows: {total_rows}")
            
            processed_data = []
            current_idx = 0
            
            # Process file in chunks
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = []
                
                for chunk in pd.read_csv(
                    file_path,
                    chunksize=self.config.chunk_size,
                    dtype_backend='pyarrow',
                    on_bad_lines='warn'
                ):
                    if not chunk.empty:
                        future = executor.submit(self._process_chunk, chunk, current_idx)
                        futures.append(future)
                        current_idx += len(chunk)
                        
                        # Log progress
                        if current_idx % (self.config.chunk_size * 10) == 0:
                            progress = (current_idx / total_rows) * 100
                            logging.info(f"Processing progress: {progress:.2f}%")
                
                # Collect results
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            processed_data.extend(result)
                    except Exception as e:
                        logging.error(f"Error collecting future result: {str(e)}")

            # Prepare response
            if not processed_data:
                return {
                    "status": "error",
                    "message": "No embeddings were generated",
                    "details": "Check logs for processing errors"
                }

            return {
                "status": "success",
                "message": f"Successfully processed {len(processed_data)} embeddings",
                "data": processed_data,
                "stats": {
                    "total_rows": total_rows,
                    "processed_embeddings": len(processed_data),
                    "success_rate": (len(processed_data) / total_rows) * 100
                }
            }

        except Exception as e:
            logging.error(f"Error in embedding generation: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": str(e),
                "details": "Unexpected error during processing"
            }