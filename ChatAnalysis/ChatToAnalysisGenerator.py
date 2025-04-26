from typing import Dict, Any, Optional, List, Union
import logging
import pandas as pd
import numpy as np
import google.generativeai as genai
import re
import os
from datetime import datetime

class ChatToAnalysisGenerator:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the analysis service with Gemini API key."""
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key not provided")
        
        self.configure_gemini()
        self.logger = logging.getLogger(__name__)
        self.setup_logger()

    def setup_logger(self) -> None:
        """Configure logging for better debugging."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def configure_gemini(self) -> None:
        """Configure Gemini with API key."""
        genai.configure(api_key=self.api_key)
        # Use Gemini 2.0 Pro for advanced analytics capabilities
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    def _get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive DataFrame information for analysis context."""
        # Basic column info
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Get basic statistics for numeric columns
        stats = {}
        for col in numeric_cols:
            stats[col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'std': float(df[col].std()),
                'skew': float(df[col].skew()) if hasattr(df[col], 'skew') else None
            }
        
        # Get value counts and unique counts for categorical columns
        categorical_stats = {}
        for col in categorical_cols:
            categorical_stats[col] = {
                'unique_count': int(df[col].nunique()),
                'top_values': df[col].value_counts().head(5).to_dict(),
                'is_binary': df[col].nunique() == 2
            }

        # Get time series information for date columns
        time_stats = {}
        for col in date_cols:
            time_stats[col] = {
                'min_date': df[col].min().strftime('%Y-%m-%d') if not pd.isna(df[col].min()) else None,
                'max_date': df[col].max().strftime('%Y-%m-%d') if not pd.isna(df[col].max()) else None,
                'date_range': (df[col].max() - df[col].min()).days if not pd.isna(df[col].min()) and not pd.isna(df[col].max()) else None
            }

        # Detect potential relationships and patterns
        correlation_matrix = None
        if len(numeric_cols) > 1:
            correlation_matrix = df[numeric_cols].corr().round(2).to_dict()

        return {
            'columns': df.columns.tolist(),
            'dtypes': {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'datetime_columns': date_cols,
            'row_count': len(df),
            'null_counts': df.isnull().sum().to_dict(),
            'sample_data': df.head(3).to_dict(),
            'numeric_stats': stats,
            'categorical_stats': categorical_stats,
            'time_stats': time_stats,
            'correlation': correlation_matrix
        }

    def _detect_analysis_type(self, query: str) -> str:
        """Detect the type of analysis needed based on the query."""
        analysis_types = {
            'regression': ['predict', 'regression', 'relationship between', 'impact of', 'affects', 'correlation'],
            'classification': ['classify', 'categorize', 'predict category', 'group', 'segment'],
            'clustering': ['cluster', 'segment', 'group similar', 'find patterns'],
            'time_series': ['time', 'trend', 'forecast', 'predict future', 'seasonal'],
            'statistical_test': ['significant', 'difference', 'test', 'hypothesis', 'compare groups'],
            'dimensionality_reduction': ['reduce dimensions', 'pca', 'feature extraction']
        }
        
        query_lower = query.lower()
        for analysis_type, keywords in analysis_types.items():
            if any(keyword in query_lower for keyword in keywords):
                return analysis_type
        
        return 'exploratory'  # Default to exploratory analysis

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
            
            # Approved ML/data analysis imports
            approved_imports = [
                'pandas', 'numpy', 're', 'math', 'statistics', 'datetime',
                'scipy', 'sklearn', 'statsmodels'
            ]
            
            # Only check for truly dangerous patterns
            dangerous_patterns = [
                r"os\.", r"sys\.", r"subprocess\.", r"shutil\.",
                r"eval\(", r"exec\(", r"__import__\(", 
                r"open\(.*,'w'\)", r"open\(.*,'a'\)", r"\.write\(",
            ]
            
            # Check for dangerous patterns
            for pattern in dangerous_patterns:
                if re.search(pattern, code):
                    self.logger.warning(f"Dangerous pattern found: {pattern}")
                    return False
                    
            # Check for unapproved imports more carefully
            import_lines = re.findall(r"^\s*(?:from|import)\s+.*", code, re.MULTILINE)
            for line in import_lines:
                is_safe = False
                for safe_lib in approved_imports:
                    if f"import {safe_lib}" in line or f"from {safe_lib}" in line:
                        is_safe = True
                        break
                if not is_safe:
                    self.logger.warning(f"Unapproved import: {line.strip()}")
                    return False
            
            # Check for required function structure
            if "def analyze_dataframe(df)" not in code:
                self.logger.warning("Required function 'analyze_dataframe(df)' not found")
                return False
                
            return True
                
        except Exception as e:
            self.logger.error(f"Code validation error: {str(e)}")
            return False

    def _execute_analysis_code(self, code: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute the analysis code safely and return results."""
        try:
            # Create a clean namespace with only required libraries
            namespace = {
                'pd': pd,
                'np': np,
                'stats': __import__('scipy.stats', fromlist=['stats']).stats,
                'df': df.copy(),  # Use a copy to prevent modification of original
                'result': None
            }
            
            # Add function to call the analyze_dataframe function
            exec_code = code + "\n\nresult = analyze_dataframe(df)"
            
            # Execute the code in the isolated namespace
            exec(exec_code, namespace)
            
            # Get the result
            result = namespace.get('result', {})
            
            # Convert pandas objects to dictionaries for JSON serialization
            processed_result = {}
            for key, value in result.items():
                if isinstance(value, pd.DataFrame):
                    processed_result[key] = value.to_dict()
                elif isinstance(value, np.ndarray):
                    processed_result[key] = value.tolist()
                elif hasattr(value, 'tolist'):
                    processed_result[key] = value.tolist()
                else:
                    processed_result[key] = value
            
            return {
                'result': processed_result,
                'intermediate_data': {}  # Simplified to avoid serialization issues
            }
            
        except Exception as e:
            self.logger.error(f"Analysis execution error: {str(e)}")
            return {
                'result': {'error': str(e)},
                'intermediate_data': {}
            }
            
    def generate_natural_language_response(self, query: str, analysis_results: Dict[str, Any], 
                                          data_info: Dict[str, Any], analysis_type: str) -> str:
        """Generate a natural language response based on the analysis results."""
        try:
            # Extract key information based on analysis type
            response_focus = {
                'regression': "Explain relationships, coefficients, model fit, and predictions",
                'classification': "Discuss classification results, accuracy, important features, and confusions",
                'clustering': "Explain discovered clusters, their characteristics, and insights",
                'time_series': "Detail trends, seasonality, forecasts, and confidence intervals",
                'statistical_test': "Interpret test results, p-values, significance, and implications",
                'dimensionality_reduction': "Explain variance explained, components, and patterns found",
                'exploratory': "Summarize key findings, distributions, and interesting insights"
            }.get(analysis_type, "Provide comprehensive answers with supporting data")
            
            result_prompt = f"""
            Based on the following data analysis results, generate a comprehensive, natural language response.
            
            User Query: {query}
            
            Analysis Type: {analysis_type}
            
            Analysis Results: {analysis_results}
            
            Data Context:
            - Total records: {data_info['row_count']}
            - Key columns: {', '.join(data_info['columns'][:10])}
            - Numeric statistics available for: {', '.join(data_info['numeric_columns'][:5])}
            
            Requirements for your response:
            1. Start with a direct answer to the user's query
            2. {response_focus}
            3. Provide supporting details with specific numbers from the results
            4. Explain any statistical concepts in simple terms
            5. Use natural, conversational language
            6. Include relevant limitations or assumptions
            7. Mention any important correlations or patterns discovered
            8. Format with clear paragraphs and structure
            
            Your response should be clear to a non-technical audience while still being statistically sound.
            """

            response = self.model.generate_content(result_prompt)
            if not response.candidates or not response.candidates[0].content.parts:
                raise ValueError("Failed to generate natural language response")
            
            return response.candidates[0].content.parts[0].text

        except Exception as e:
            self.logger.error(f"Error generating natural language response: {str(e)}")
            return str(analysis_results['result'])  # Fallback to raw result

    def generate_analysis(self, df: pd.DataFrame, user_query: str) -> Dict[str, Any]:
        """Generate advanced analysis based on user query."""
        try:
            # Get comprehensive data information
            data_info = self._get_data_info(df)
            
            # Detect analysis type from query
            analysis_type = self._detect_analysis_type(user_query)
            self.logger.info(f"Detected analysis type: {analysis_type}")
            
            # Create simplified prompt for analysis
            prompt = r"""
            Generate Python analysis code for a dataframe based on this query: "{user_query}"

            Analysis type: {analysis_type}

            DataFrame info:
            - Columns: {', '.join(data_info['columns'][:10])}
            - Row count: {data_info['row_count']}
            - Numeric columns: {', '.join(data_info['numeric_columns'][:10])}
            - Categorical columns: {', '.join(data_info['categorical_columns'][:10])}

            IMPORTANT:
            1. Define a function named exactly `analyze_dataframe(df)` that returns a dictionary
            2. Use only these libraries: pandas (as pd), numpy (as np), scipy.stats
            3. NO sklearn or other ML libraries
            4. NO os, sys, eval, exec or file operations
            5. NO example data creation - work with the input df parameter
            6. Store all important findings in the result dictionary
            7. Include proper error handling with try/except

            Here's the template to follow:

            ```python
            import pandas as pd
            import numpy as np
            from scipy import stats

            def analyze_dataframe(df):
                result = {}
                try:
                    # 1. Perform data preparation
                    
                    # 2. Calculate summary statistics

                    # 3. Find relationships and patterns

                    # 4. Identify key insights
                    
                    # Return your findings as a dictionary
                    return result
                except Exception as e:
                    result['error'] = str(e)
                    return result
            ```

            Return ONLY code inside triple backticks with no additional text.
            """

            # Generate and validate analysis
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Generate response
                    response = self.model.generate_content(prompt)
                    response_text = response.candidates[0].content.parts[0].text
                    code = self._extract_python_code(response_text)
                    
                    if not self._validate_code(code):
                        raise ValueError("Invalid or unsafe code generated")

                    # Execute analysis and get results
                    execution_results = self._execute_analysis_code(code, df)
                    
                    # Generate natural language response
                    natural_response = self.generate_natural_language_response(
                        user_query,
                        execution_results,
                        data_info,
                        analysis_type
                    )

                    return {
                        'status': 'success',
                        'analysis_type': analysis_type,
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
            self.logger.error(f"Analysis generation failed: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e),
                'type': str(type(e).__name__)
            }
    
    def _get_analysis_standards(self, analysis_type: str) -> str:
        """Get specific standards for different types of analysis."""
        standards = {
            'regression': """
            - Handle continuous target variable prediction 
            - Perform feature selection or importance analysis
            - Evaluate model with metrics like R², RMSE, and MAE
            - Consider both linear and non-linear approaches
            - Check for multicollinearity and outliers
            - Store model coefficients/importance in intermediate_data
            """,
            
            'classification': """
            - Ensure proper encoding of categorical features and target
            - Balance classes if needed
            - Use appropriate metrics (accuracy, precision, recall, F1)
            - Compare performance against a baseline
            - Store confusion matrix and classification report in intermediate_data
            - Include feature importance analysis
            """,
            
            'clustering': """
            - Scale numeric features appropriately
            - Determine optimal number of clusters
            - Characterize each cluster with descriptive statistics
            - Validate clustering quality with silhouette score or similar
            - Store cluster assignments and centroids in intermediate_data
            """,
            
            'time_series': """
            - Check for trend, seasonality, and stationarity
            - Handle missing values with appropriate interpolation
            - Use train/test split with time-aware validation
            - Evaluate forecasts with metrics like MAPE and RMSE
            - Store decomposition components in intermediate_data
            """,
            
            'statistical_test': """
            - Check assumptions before applying tests
            - Use appropriate tests (t-test, ANOVA, chi-square)
            - Calculate effect sizes, not just p-values
            - Store test statistics and p-values in intermediate_data
            - Use appropriate corrections for multiple comparisons
            """,
            
            'dimensionality_reduction': """
            - Scale features before applying PCA or other techniques
            - Determine optimal number of components 
            - Evaluate variance explained by components
            - Store transformed data and loadings in intermediate_data
            """,
            
            'exploratory': """
            - Calculate summary statistics (mean, median, std)
            - Check for correlations between variables
            - Identify outliers and unusual patterns
            - Create segmentations if appropriate
            - Store key findings in intermediate_data
            """
        }
        
        return standards.get(analysis_type, "Perform comprehensive analysis appropriate to the query")