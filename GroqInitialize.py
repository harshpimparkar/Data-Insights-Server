import os
import logging
from flask import request, jsonify
from dotenv import load_dotenv
from langdetect import detect
from groq import Groq
from groq._base_client import SyncHttpxClientWrapper
# Load environment variables
load_dotenv()

# Groq API key setup
GROQ_API_KEY = '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'
if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY is not set in the environment variables.")

# Initialize Groq client
http_client = SyncHttpxClientWrapper()
client = Groq(api_key=GROQ_API_KEY , http_client=http_client)

# Groq model parameters
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 1024

# Prompt template
PROMPT_TEMPLATE = """
Question: {query}
You are a data scientist. The conversation should be strictly in English. You are supposed to help the user with any questions they have. Analyze the provided data and generate detailed text for proper display.
"""

def initialize_groq_api(chat_request):
    """
    Handle requests to the Groq API for chat completions.
    """
    try:
        # Validate input type
        if not isinstance(chat_request, dict):
            logging.error("Invalid request format: Expected a dictionary.")
            return {"error": "Invalid request format. Please provide a valid dictionary."}, 400

        # Extract messages from the chat request
        messages = chat_request.get('messages', [])
        if not messages:
            logging.error("No messages provided in the chat request.")
            return {"error": "No messages provided in the request."}, 400

        logging.info(f"Received messages: {messages}")

        # Build the input for the Groq API
        full_messages = [{"role": "system", "content": PROMPT_TEMPLATE}]
        for msg in messages:
            if "content" in msg and msg["content"].strip():
                prompt = PROMPT_TEMPLATE.format(query=msg["content"])
                full_messages[0]["content"] = prompt
                msg["role"] = "user"
                full_messages.append(msg)

        # Send the chat completion request to Groq API
        response = client.chat.completions.create(
            messages=full_messages,
            model=DEFAULT_MODEL,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS
        )

        # Extract assistant response and token usage
        assistant_message = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        logging.info(f"Assistant response: {assistant_message}")

        return {
            "message": assistant_message,
            "tokens_used": tokens_used
        }, 200

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        return {"error": f"An error occurred: {str(e)}"}, 500


    """Generate a structured prompt for the LLM based on insights"""
    formatted_insights = "\n".join([
        f"- {key.capitalize()}: {str(value)}"
        for key, value in insights_context.items()
        if value and not isinstance(value, dict)
    ])
    
    return {
        "messages": [{
            "content": VIZ_PROMPT_TEMPLATE.format(insights=formatted_insights),
            "role": "user"
        }]
    }
        
def compress_llm_context(matches, query, max_tokens=6000):
    """
    Compresses context data for LLM while preserving important information.
    
    Args:
        matches: List of processed matches from vector search
        query: Original user query
        max_tokens: Maximum allowed tokens (default: 6000)
    
    Returns:
        dict: Compressed context with token estimation
    """
    def estimate_tokens(text):
        """Estimate number of tokens in text (approximation)"""
        return len(text.split()) + len(text) // 4

    def extract_relevant_content(match):
        """Extract and clean relevant content from a match"""
        content = match.get('metadata', {}).get('content', '')
        score = match.get('score', 0)
        return {'content': content, 'score': score}

    try:
        compressed_matches = []
        total_tokens = 0
        system_tokens = 500  # Reserve tokens for system prompt
        
        # Sort matches by score
        sorted_matches = sorted(
            [extract_relevant_content(m) for m in matches],
            key=lambda x: x['score'],
            reverse=True
        )

        # Compress and add matches while staying within token limit
        for match in sorted_matches:
            content = match['content']
            
            # Basic content compression
            content = ' '.join(content.split())  # Remove extra whitespace
            content = content.replace('\n', ' ')
            
            # Estimate tokens for this content
            content_tokens = estimate_tokens(content)
            
            # Check if adding this content would exceed token limit
            if total_tokens + content_tokens + system_tokens <= max_tokens:
                compressed_matches.append({
                    'content': content,
                    'score': match['score']
                })
                total_tokens += content_tokens
            else:
                break

        return {
            'compressed_matches': compressed_matches,
            'total_tokens': total_tokens + system_tokens,
            'original_count': len(matches),
            'compressed_count': len(compressed_matches)
        }

    except Exception as e:
        logging.error(f"Error in compress_llm_context: {str(e)}")
        return {
            'compressed_matches': [],
            'total_tokens': 0,
            'original_count': len(matches),
            'compressed_count': 0,
            'error': str(e)
        }