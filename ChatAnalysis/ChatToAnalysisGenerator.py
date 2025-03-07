from typing import Dict, Any, Optional
import logging
import pandas as pd
import google.generativeai as genai
import re
import os

class ChatToAnalysisGenerator:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the analysis service with Gemini API key."""
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key not provided")
        
        self.configure_gemini()
        self.logger = logging.getLogger(__name__)

    def configure_gemini(self) -> None:
        """Configure Gemini with API key."""
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    def _get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive DataFrame information for analysis context."""
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Get basic statistics for numeric columns
        stats = {}
        for col in numeric_cols:
            stats[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'min': df[col].min(),
                'max': df[col].max()
            }
        
        # Get value counts for categorical columns (limited to top 5)
        categorical_stats = {}
        for col in df.select_dtypes(exclude=['number']).columns:
            categorical_stats[col] = df[col].value_counts().head(5).to_dict()

        return {
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'numeric_columns': numeric_cols,
            'categorical_columns': df.select_dtypes(exclude=['number']).columns.tolist(),
            'row_count': len(df),  # This is correct
            'null_counts': df.isnull().sum().to_dict(),
            'sample_data': df.head(5).to_dict(),  # This is causing confusion
            'numeric_stats': stats,
            'categorical_stats': categorical_stats
        }

    def _extract_python_code(self, response_text: str) -> str:
        """Extract Python code from Gemini response."""
        code_pattern = r"```python(.*?)```"
        matches = re.findall(code_pattern, str(response_text), re.DOTALL)
        return matches[0].strip() if matches else ""

    def _validate_code(self, code: str) -> bool:
        """Validate the generated analysis code."""
        if not code:
            return False
        try:
            compile(code, '<string>', 'exec')
            # Check for dangerous operations
            dangerous_patterns = [
                r"os\.", r"sys\.", r"subprocess\.", r"eval\(",
                r"exec\(", r"import\s+(?!pandas|numpy|re|math|statistics|datetime)"
            ]
            if any(re.search(pattern, code) for pattern in dangerous_patterns):
                self.logger.warning(f"Dangerous pattern found in code: {code}")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Code validation error: {str(e)}")
            return False

    def _execute_analysis_code(self, code: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute the analysis code and return both the result and intermediate data."""
        try:
            # Extract just the function part, removing any example data creation
            # This is a simple regex to remove data creation and focus on the analysis function
            code_parts = re.split(r'(?:^|\n)(?:# Example|data =|df =)', code)
            analysis_code = code_parts[0]
            
            # Import numpy directly
            import numpy as np
            
            namespace = {
                'pd': pd,
                'np': np,  # Use the imported numpy module, not pd.np
                'df': df,
                'result': None,
                'intermediate_data': {}
            }
            
            # Add code to store intermediate results
            modified_code = analysis_code + "\n"
            
            # If there's a function definition, add code to call it
            if 'def analyze' in modified_code:
                modified_code += "\nresult = analyze_dataframe(df)\n"
            
            modified_code += "intermediate_data['final_result'] = result"
            
            exec(modified_code, namespace)
            
            return {
                'result': namespace.get('result'),
                'intermediate_data': namespace.get('intermediate_data', {})
            }
        except Exception as e:
            self.logger.error(f"Analysis execution error: {str(e)}")
            raise

    def generate_natural_language_response(self, query: str, analysis_results: Dict[str, Any], data_info: Dict[str, Any]) -> str:
        """Generate a natural language response based on the analysis results."""
        try:
            result_prompt = f"""
            Based on the following data analysis results, generate a comprehensive, natural language response.
            
            User Query: {query}
            
            Analysis Results: {analysis_results}
            
            Data Context:
            - Total records in the full dataset: {data_info['row_count']}
            - Relevant statistics: {data_info['numeric_stats']}
            
            Requirements for the response:
            1. Start with a direct answer to the user's query
            2. Be clear about whether you're referring to the full dataset or any subset
            3. Provide supporting details and context
            4. Include relevant numbers and statistics
            5. Explain any important patterns or insights
            6. Use natural, conversational language
            7. Be concise but thorough
            8. Highlight any limitations or assumptions
            
            Format the response in a clear, readable way with appropriate paragraphs.
            """

            response = self.model.generate_content(result_prompt)
            if not response.candidates or not response.candidates[0].content.parts:
                raise ValueError("Failed to generate natural language response")
            
            return response.candidates[0].content.parts[0].text

        except Exception as e:
            self.logger.error(f"Error generating natural language response: {str(e)}")
            return str(analysis_results['result'])  # Fallback to raw result

    def generate_analysis(self, df: pd.DataFrame, user_query: str) -> Dict[str, Any]:
        """Generate analysis based on user query."""
        try:
            # Get comprehensive data information
            data_info = self._get_data_info(df)
            
            # Create focused prompt for analysis
            prompt = f"""
           You are an expert data analyst. Generate precise Python code to analyze this DataFrame based on the query.

            ### User Query:
            {user_query}

            ### DataFrame Information:
            - Columns: {', '.join(data_info['columns'])}
            - Row count: {data_info['row_count']}
            - Data types: {data_info['dtypes']}
            - Sample data: {data_info['sample_data']}
            - Numeric column statistics: {data_info['numeric_stats']}
            - Category distributions: {data_info['categorical_stats']}

            ### Code Requirements:
            1.**Define your analysis in a function named exactly `analyze_dataframe(df)`
            2. **Ensure column names match the DataFrame.**  
            3. Perform the analysis using **pandas/numpy only**.  
            4. **Check for missing values and handle them appropriately.**  
            5. Store the **final result** in `result`.  
            6. Store **any intermediate results** in `intermediate_data`.  
            7. **Include comments explaining each step** for better understanding.  
            8. **NEVER use os, sys, subprocess, eval, exec, or unsafe imports.**  
            9. **DO NOT create example or dummy data. Your code will run against the actual dataframe.**
            10. **Return code inside triple backticks (` ```python ``` `) only.** 

            ### Example Code Format:
            ```python
            import pandas as pd
            import numpy as np

            # Ensure column exists
            if 'column_name' in df.columns:
                result = df['column_name'].mean()
                intermediate_data['summary'] = df['column_name'].describe()
            else:
                result = "Error: Column not found"

                        ```
            """

            # Generate and validate analysis
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Generate response
                    response = self.model.generate_content(prompt)
                    if not response.candidates or not response.candidates[0].content.parts:
                        raise ValueError("Invalid Gemini response")
                    
                    response_text = response.candidates[0].content.parts[0].text
                    code = self._extract_python_code(response_text)
                    
                    if not self._validate_code(code):
                        raise ValueError("Invalid code generated")

                    # Execute analysis and get results
                    execution_results = self._execute_analysis_code(code, df)
                    
                    # Generate natural language response
                    natural_response = self.generate_natural_language_response(
                        user_query,
                        execution_results,
                        data_info
                    )

                    return {
                        'status': 'success',
                        'result': execution_results['result'],
                        'intermediate_data': execution_results['intermediate_data'],
                        'answer': natural_response,
                        'code': code
                    }

                except Exception as e:
                    self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt == max_retries - 1:
                        raise

            raise RuntimeError("Failed to generate analysis after multiple attempts")

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'type': str(type(e).__name__)
            }