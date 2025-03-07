def format_context_from_matches(matches, query):
    """
    Format retrieved matches into natural language context for the LLM.
    Converts vector search results into a readable format.
    """
    if not matches:
        return "No relevant data points were found in the database."
    
    context_parts = ["Here are the relevant data points from the database:"]
    for i, match in enumerate(matches, 1):
        metadata = match.get('metadata', {})
        data_points = [f"{key}: {value if not isinstance(value, (int, float)) else f'{value:.2f}' }"
                       for key, value in metadata.items()]
        data_description = ", ".join(data_points)
        similarity = f"(confidence score: {match['score']:.2f})"
        context_parts.append(f"\n{i}. {data_description} {similarity}")
    
    context_parts.append(f"\nThese {len(matches)} data points snippets were retrieved based on their relevance to the query: '{query}'")
    return "\n".join(context_parts)

def generate_llm_prompt(query, context):
    """
    Generate a comprehensive prompt for the LLM using the query and context.
    """
    return {
        "messages": [
            {"role": "system", "content": "You are a data analyst assistant. ... [rules for analysis]"},
            {"role": "user", "content": f"""
            Analysis Request:
            Question: {query}

            Available Data:
            {context}

            Please provide a detailed analysis based on these data points and the query. Include specific numbers about the data and explain any patterns or insights you notice. DO not use confidence in your answer. use bullet points. Highlight titles or headers of the data points.
            """}
        ]
    }
