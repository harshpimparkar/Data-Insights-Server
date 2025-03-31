# import os
# import logging
# import pandas as pd
# import numpy as np
# import json
# import google.generativeai as genai
# import re
# from typing import Dict, Any, List, Optional

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# class SmartDataExtractor:
#     def __init__(self, model=None):
#         """Initialize with an existing Gemini model instance."""
#         self.model = model
#         self.logger = logging.getLogger(__name__)
    
#     def identify_relevant_columns(self, df: pd.DataFrame, user_query: str) -> List[str]:
#         """Use the provided model to identify columns relevant to the user query."""
#         if self.model is None:
#             # Fallback to basic extraction if no model provided
#             self.logger.warning("No model provided, falling back to basic column extraction")
#             return self._basic_column_extraction(df, user_query)
        
#         # Create a prompt to analyze the query and dataset schema
#         columns_info = {
#             col: {
#                 "dtype": str(df[col].dtype),
#                 "sample_values": df[col].dropna().sample(min(5, len(df))).tolist() if len(df) > 0 else [],
#                 "unique_count": df[col].nunique(),
#                 "description": self._infer_column_description(col, df[col])
#             } for col in df.columns
#         }
        
#         prompt = f"""
#         Analyze this user query about a dataset and identify which columns are most relevant for answering it.
        
#         User Query: "{user_query}"
        
#         Dataset Columns:
#         {json.dumps(columns_info, indent=2)}
        
#         Return a JSON object with two keys:
#         1. "relevant_columns": A list of column names that are directly relevant to the query
#         2. "reasoning": Brief explanation of why each column was selected
        
#         Only include columns that are clearly relevant to answering the query.
#         """
        
#         try:
#             self.logger.info("Sending column identification request to Gemini")
#             response = self.model.generate_content(prompt)
#             response_text = response.candidates[0].content.parts[0].text
#             self.logger.debug(f"Received response: {response_text[:200]}...")
            
#             # Extract JSON from response
#             json_pattern = r"```json(.*?)```"
#             matches = re.findall(json_pattern, response_text, re.DOTALL)
#             if matches:
#                 json_str = matches[0]
#             else:
#                 # Try to find JSON without code blocks
#                 json_str = re.search(r"\{.*\}", response_text, re.DOTALL)
#                 if json_str:
#                     json_str = json_str.group(0)
#                 else:
#                     json_str = response_text
            
#             result = json.loads(json_str)
#             relevant_cols = result.get("relevant_columns", [])
#             self.logger.info(f"Identified {len(relevant_cols)} relevant columns: {relevant_cols}")
#             return relevant_cols
            
#         except Exception as e:
#             # Fallback to basic extraction if LLM fails
#             self.logger.error(f"Error in LLM column extraction: {str(e)}")
#             return self._basic_column_extraction(df, user_query)
    
#     def _basic_column_extraction(self, df: pd.DataFrame, user_query: str) -> List[str]:
#         """Simple fallback method to extract relevant columns based on string matching."""
#         query_lower = user_query.lower()
#         potentially_relevant_cols = [
#             col for col in df.columns 
#             if col.lower() in query_lower or col.lower().replace('_', ' ') in query_lower
#         ]
#         self.logger.info(f"Basic extraction found {len(potentially_relevant_cols)} columns")
#         return potentially_relevant_cols if potentially_relevant_cols else df.columns.tolist()
    
#     def _infer_column_description(self, column_name: str, series: pd.Series) -> str:
#         """Generate a simple description of what the column likely represents."""
#         if pd.api.types.is_datetime64_any_dtype(series):
#             return "datetime values"
#         elif pd.api.types.is_bool_dtype(series):
#             return "boolean values"
#         elif pd.api.types.is_numeric_dtype(series):
#             if column_name.lower() in ['id', 'index', 'key']:
#                 return "identifier values"
#             elif any(term in column_name.lower() for term in ['price', 'cost', 'revenue', 'sales', 'income']):
#                 return "monetary values"
#             elif any(term in column_name.lower() for term in ['count', 'num', 'qty', 'quantity']):
#                 return "count values"
#             else:
#                 return "numeric values"
#         else:
#             if series.nunique() < 10 and series.nunique() / len(series) < 0.1:
#                 return "categorical values"
#             elif any(term in column_name.lower() for term in ['name', 'title', 'description']):
#                 return "text description"
#             else:
#                 return "string values"
    
#     def extract_relevant_data(self, df: pd.DataFrame, user_query: str) -> Dict[str, Any]:
#         """Extract query-relevant data from the DataFrame."""
#         self.logger.info(f"Extracting relevant data for query: {user_query}")
#         # Identify relevant columns using LLM or fallback
#         relevant_columns = self.identify_relevant_columns(df, user_query)
        
#         # Start with basic dataset info
#         data_info = {
#             'all_columns': df.columns.tolist(),
#             'row_count': len(df),
#             'relevant_columns': relevant_columns
#         }
        
#         # If no relevant columns identified, use all columns
#         if not relevant_columns:
#             self.logger.warning("No relevant columns identified, using all columns")
#             relevant_columns = df.columns.tolist()
        
#         # For each relevant column, extract appropriate information based on type
#         for col in relevant_columns:
#             if col not in df.columns:
#                 self.logger.warning(f"Column '{col}' not found in DataFrame")
#                 continue
                
#             # Get column-specific information
#             if pd.api.types.is_numeric_dtype(df[col]):
#                 # For numeric columns, provide detailed statistics
#                 data_info[f'{col}_stats'] = {
#                     'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
#                     'median': float(df[col].median()) if not pd.isna(df[col].median()) else None,
#                     'std': float(df[col].std()) if not pd.isna(df[col].std()) else None,
#                     'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
#                     'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
#                 }
                
#                 # Provide histogram-like distribution (10 bins)
#                 try:
#                     hist, bin_edges = np.histogram(df[col].dropna(), bins=10)
#                     data_info[f'{col}_distribution'] = {
#                         'bins': [float(b) for b in bin_edges],
#                         'counts': [int(h) for h in hist]
#                     }
#                 except Exception as e:
#                     self.logger.warning(f"Error creating histogram for {col}: {str(e)}")
#             else:
#                 # For categorical/text columns, provide value counts
#                 try:
#                     value_counts = df[col].value_counts().head(10)
#                     data_info[f'{col}_distribution'] = {
#                         'values': value_counts.index.tolist(),
#                         'counts': value_counts.values.tolist()
#                     }
#                 except Exception as e:
#                     self.logger.warning(f"Error getting value counts for {col}: {str(e)}")
            
#             # Include sample values from across the distribution
#             try:
#                 if df[col].nunique() > 1:
#                     # Try to get diverse samples across distribution
#                     if pd.api.types.is_numeric_dtype(df[col]):
#                         # For numeric, sample from different quantiles
#                         quantiles = df[col].quantile([0.1, 0.3, 0.5, 0.7, 0.9]).tolist()
#                         data_info[f'{col}_samples'] = quantiles
#                     else:
#                         # For categorical, get most frequent values
#                         data_info[f'{col}_samples'] = df[col].value_counts().head(5).index.tolist()
#                 else:
#                     data_info[f'{col}_samples'] = [df[col].iloc[0]]
#             except Exception as e:
#                 self.logger.warning(f"Error getting sample values for {col}: {str(e)}")
        
#         # Include correlations between relevant numeric columns
#         numeric_relevant = [c for c in relevant_columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
#         if len(numeric_relevant) > 1:
#             corr_matrix = df[numeric_relevant].corr()
            
#             # Extract significant correlations (absolute value > 0.5)
#             significant_correlations = []
#             for i in range(len(numeric_relevant)):
#                 for j in range(i+1, len(numeric_relevant)):
#                     col1, col2 = numeric_relevant[i], numeric_relevant[j]
#                     corr_value = corr_matrix.loc[col1, col2]
#                     if abs(corr_value) > 0.5:
#                         significant_correlations.append({
#                             'columns': [col1, col2],
#                             'correlation': float(corr_value)
#                         })
            
#             data_info['significant_correlations'] = significant_correlations
#             self.logger.info(f"Identified {len(significant_correlations)} significant correlations")
        
#         return data_info