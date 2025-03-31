# from typing import Dict, Any, Optional
# import logging
# import pandas as pd
# import google.generativeai as genai
# import re
# import os
# import ast

# class ChatToAnalysisGenerator:
#     def __init__(self, api_key: Optional[str] = None):
#         self.api_key = api_key or os.getenv('GEMINI_API_KEY')
#         if not self.api_key:
#             raise ValueError("Gemini API key not provided")
        
#         self.configure_gemini()
#         self.logger = logging.getLogger(__name__)

#     def configure_gemini(self) -> None:
#         genai.configure(api_key=self.api_key)
#         self.model = genai.GenerativeModel("gemini-2.0-flash")

#     def _get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
#         numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
#         stats = {col: {'mean': df[col].mean(), 'median': df[col].median(), 'min': df[col].min(), 'max': df[col].max()} for col in numeric_cols}
#         categorical_stats = {col: df[col].value_counts().head(5).to_dict() for col in df.select_dtypes(exclude=['number']).columns}
#         return {
#             'columns': df.columns.tolist(),
#             'dtypes': df.dtypes.to_dict(),
#             'row_count': len(df),
#             'null_counts': df.isnull().sum().to_dict(),
#             'numeric_stats': stats,
#             'categorical_stats': categorical_stats
#         }

#     def _extract_python_code(self, response_text: str) -> str:
#         code_pattern = r"```python(.*?)```"
#         matches = re.findall(code_pattern, response_text, re.DOTALL)
#         return matches[0].strip() if matches else ""

#     def _validate_code(self, code: str, data_info: Dict[str, Any]) -> bool:
#         if not code:
#             return False
#         try:
#             tree = ast.parse(code)
#             has_result_assignment = any(
#                 isinstance(node, ast.Assign) and any(
#                     isinstance(target, ast.Name) and target.id == 'result' for target in node.targets
#                 ) for node in tree.body
#             )
#             if not has_result_assignment:
#                 self.logger.error("Generated code does not assign to 'result'")
#                 return False
            
#             used_columns = set(re.findall(r"df\['(.*?)'\]", code))
#             if not used_columns.issubset(set(data_info['columns'])):
#                 self.logger.error("Generated code references non-existent columns")
#                 return False
            
#             dangerous_patterns = [r"os\.", r"sys\.", r"subprocess\.", r"eval\(", r"exec\("]
#             if any(re.search(pattern, code) for pattern in dangerous_patterns):
#                 self.logger.error("Code contains unsafe operations")
#                 return False
            
#             return True
#         except SyntaxError as e:
#             self.logger.error(f"Code syntax error: {str(e)}")
#             return False

#     def _execute_analysis_code(self, code: str, df: pd.DataFrame) -> Dict[str, Any]:
#         try:
#             namespace = {'pd': pd, 'df': df, 'result': None, 'intermediate_data': {}}
#             modified_code = code + "\nintermediate_data['final_result'] = result"
#             exec(modified_code, namespace)
#             return {'result': namespace.get('result'), 'intermediate_data': namespace.get('intermediate_data', {})}
#         except Exception as e:
#             self.logger.error(f"Execution error: {str(e)}")
#             return {'status': 'error', 'message': str(e), 'code': code}

#     def generate_analysis(self, df: pd.DataFrame, user_query: str) -> Dict[str, Any]:
#         try:
#             data_info = self._get_data_info(df)
#             prompt = f"""
#             You are an expert data analyst. Generate **precise** Python code to analyze this DataFrame based on the user query:
            
#             Query: {user_query}
            
#             Data Overview:
#             - Columns: {data_info['columns']}
#             - Row count: {data_info['row_count']}
#             - Data types: {data_info['dtypes']}
#             - Numeric column stats: {data_info['numeric_stats']}
#             - Categorical distributions: {data_info['categorical_stats']}
            
#             **Ensure:**
#             1. The generated code only references columns present in the dataset.
#             2. Handles missing values appropriately.
#             3. Stores final computed value in 'result'.
#             4. Uses pandas/numpy (no eval, exec, os, sys, subprocess).
#             5. Formats response as:
#             ```python
#             # Your analysis code
#             ```
#             """
            
#             for attempt in range(3):
#                 response = self.model.generate_content(prompt)
#                 response_text = response.candidates[0].content.parts[0].text if response.candidates else ""
#                 code = self._extract_python_code(response_text)
                
#                 if self._validate_code(code, data_info):
#                     return self._execute_analysis_code(code, df)
#                 self.logger.warning(f"Attempt {attempt + 1} failed: Invalid code")
            
#             raise RuntimeError("Failed to generate valid analysis code after multiple attempts")
        
#         except Exception as e:
#             return {'status': 'error', 'message': str(e), 'type': str(type(e).__name__)}
