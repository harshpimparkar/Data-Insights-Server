import pandas as pd
import google.generativeai as genai
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import re
import numpy as np
from typing import Optional, Tuple, Dict, Any, List

class VisualizationGenerator:
    def __init__(self, api_key: str):
        """Initialize the visualization generator."""
        self.logger = self._setup_logging()
        self.configure_gemini(api_key)
        plt.switch_backend('Agg')
        
    @staticmethod
    def _setup_logging() -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

    def configure_gemini(self, api_key: str) -> None:
        """Configure Gemini with the provided API key."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess DataFrame to ensure numeric columns are properly typed.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        processed_df = df.copy()
        
        for column in processed_df.columns:
            # Try to convert to numeric, set as category if fails
            try:
                processed_df[column] = pd.to_numeric(processed_df[column], errors='raise')
            except (ValueError, TypeError):
                processed_df[column] = processed_df[column].astype('category')
                
        return processed_df

    def _modify_seaborn_code(self, code: str) -> str:
        """
        Modify Seaborn plotting code to handle deprecation warnings.
        
        Args:
            code (str): Original visualization code
            
        Returns:
            str: Modified code with proper Seaborn configuration
        """
        # Pattern to match sns plotting functions with palette but without hue
        pattern = r'sns\.(bar|box|violin|strip|swarm|point|cat)plot\((.*?)palette\s*=\s*([^,\)]+)(.*?)\)'
        
        def replace_palette(match):
            func = match.group(1)
            args = match.group(2)
            palette = match.group(3)
            rest = match.group(4)
            
            # Check if hue is already set
            if 'hue=' in args or 'hue=' in rest:
                return f'sns.{func}plot({args}palette={palette}{rest})'
            
            # Check if data parameter is present
            if 'data=' not in args and 'data=' not in rest:
                # No data parameter, don't add hue
                return f'sns.{func}plot({args}palette={palette}{rest})'
            
            # Extract x variable if present
            x_match = re.search(r'x\s*=\s*["\']?([^,\'"]+)["\']?', args + rest)
            if x_match:
                x_var = x_match.group(1)
                # Make sure x_var is a column name, not a computed value like counts.index
                if '.' in x_var or '[' in x_var:
                    # This is likely a computed value, don't add hue
                    return f'sns.{func}plot({args}palette={palette}{rest})'
                
                # Add hue parameter and set legend=False
                if 'legend=' not in args and 'legend=' not in rest:
                    return f'sns.{func}plot({args}hue="{x_var}", legend=False, palette={palette}{rest})'
                else:
                    return f'sns.{func}plot({args}hue="{x_var}", palette={palette}{rest})'
            
            return f'sns.{func}plot({args}palette={palette}{rest})'
        
        modified_code = re.sub(pattern, replace_palette, code)
        
        # Add type conversion for numeric columns
        if 'df = pd.DataFrame' in modified_code:
            modified_code = "df = preprocess_dataframe(df)\n" + modified_code
        
        return modified_code

    def _extract_python_code(self, response_text: str) -> str:
        """Extract and clean Python code from Gemini response."""
        code_pattern = r"```python(.*?)```"
        matches = re.findall(code_pattern, str(response_text), re.DOTALL)
        
        if matches:
            code = matches[0].strip()
        else:
            code_lines = []
            for line in str(response_text).split('\n'):
                if any(term in line for term in ['import', 'plt.', '=', 'pd.', 'sns.']) or \
                   line.strip().startswith('#'):
                    code_lines.append(line)
            code = '\n'.join(code_lines)
        
        # Remove plt.show() calls and modify Seaborn code
        code = code.replace('plt.show()', '')
        code = self._modify_seaborn_code(code.strip())
        
        # Add preprocessing function definition
        preprocess_func = """
def preprocess_dataframe(df):
    processed_df = df.copy()
    for column in processed_df.columns:
        try:
            processed_df[column] = pd.to_numeric(processed_df[column], errors='raise')
        except (ValueError, TypeError):
            processed_df[column] = processed_df[column].astype('category')
    return processed_df
"""
        return preprocess_func + "\n" + code

    def _is_code_safe(self, code_string: str) -> bool:
        """Check if the code is safe to execute."""
        forbidden_terms = ['exec(', 'eval(', 'os.system', 'os.popen', 'subprocess']
        return not any(term in code_string.lower() for term in forbidden_terms)

    def _execute_visualization_code(self, code_string: str, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Execute the visualization code safely."""
        try:
            namespace = {
                'pd': pd,
                'plt': plt,
                'sns': sns,
                'np': np,
                'df': df
            }
            
            exec(code_string, namespace)
            
            self.logger.info("Visualization code executed successfully!")
            return True, namespace
            
        except Exception as e:
            self.logger.error(f"Error executing visualization code: {str(e)}")
            return False, {}

    def generate_visualization(self, 
                        df: pd.DataFrame, 
                        insight_text: str, dataset_name: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Generate and execute visualization based on insight text with improved error reporting."""
        try:
            # Preprocess the DataFrame
            df = self._preprocess_dataframe(df)
            
            response = self.model.generate_content(insight_text)
            viz_prompt = self._create_visualization_prompt(insight_text, df, dataset_name)
            code = self._extract_python_code(response.text)
            self.logger.info("Generated code extracted successfully")
            self.logger.info(f"Generated code:\n{code}")
            
            if not self._is_code_safe(code):
                self.logger.warning("Code contains potentially unsafe operations")
                return False, code, {}
                
            success, namespace = self._execute_visualization_code(code, df)
            return success, code, namespace
            
        except Exception as e:
            self.logger.error(f"Error in visualization generation: {str(e)}", exc_info=True)
            return False, "", {}