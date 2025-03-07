from typing import Dict, Any, Optional, Tuple
import logging
import base64
from io import BytesIO  
import matplotlib.pyplot as plt
import pandas as pd
import google.generativeai as genai
import seaborn as sns
import re
import matplotlib
matplotlib.use('Agg')
import os
import traceback

class ChatToVisGenerator:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the visualization service with Gemini API key."""
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key not provided")
        
        self.configure_gemini()
        self.logger = logging.getLogger(__name__)
        plt.switch_backend('Agg')

    def configure_gemini(self) -> None:
        """Configure Gemini with API key."""
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    def _get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive DataFrame information for visualization context."""
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
            'dtypes': {str(k): str(v) for k, v in df.dtypes.to_dict().items()},  # Convert to string for JSON serialization
            'numeric_columns': numeric_cols,
            'categorical_columns': df.select_dtypes(exclude=['number']).columns.tolist(),
            'row_count': len(df),
            'null_counts': df.isnull().sum().to_dict(),
            'numeric_stats': stats,
            'categorical_stats': categorical_stats
        }
    
    def _extract_python_code(self, response_text: str) -> str:
        """Extract Python code from Gemini response."""
        code_pattern = r"```python\s*(.*?)\s*```"
        matches = re.findall(code_pattern, response_text, re.DOTALL)
        return matches[0].strip() if matches else response_text.strip()

    def _validate_code(self, code: str) -> bool:
        """Validate the generated visualization code."""
        try:
            compile(code, '<string>', 'exec')
            return True
        except Exception:
            return False

    def _execute_visualization_code(self, code: str, df: pd.DataFrame) -> bool:
        """Execute the visualization code with enhanced error handling."""
        try:
            # Extract just the visualization part, removing any example data creation
            code_parts = re.split(r'(?:^|\n)(?:# Example|data =|df =)', code)
            vis_code = code_parts[0]
            
            # Ensure we have imports
            if not "import matplotlib.pyplot as plt" in vis_code:
                vis_code = "import matplotlib.pyplot as plt\nimport seaborn as sns\n" + vis_code
                
            # Ensure we're not replacing the original dataframe
            vis_code = re.sub(r'df\s*=\s*pd\.DataFrame', 'df_example = pd.DataFrame', vis_code)
            vis_code = re.sub(r'plt\.savefig\([^)]*\)', '# plt.savefig() removed', vis_code)
            namespace = {'pd': pd, 'plt': plt, 'sns': sns, 'df': df}
            exec(vis_code, namespace)
            
            return True
        except Exception as e:
            error_details = traceback.format_exc()
            self.logger.error(f"Visualization execution error:\n{error_details}")
            return False

    def _save_figures_to_base64(self) -> Dict[str, Dict[str, str]]:
        """Save all matplotlib figures to base64 format."""
        visualization_data = {}
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            buf = BytesIO()
            try:
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
                buf.seek(0)
                visualization_data[f'visualization_{fig_num}'] = {
                    'type': 'image/png',
                    'data': base64.b64encode(buf.getvalue()).decode('utf-8')
                }
            except Exception as e:
                self.logger.error(f"Error saving figure {fig_num}: {e}")
            finally:
                buf.close()
            plt.close(fig)  # Close figure after processing
        return visualization_data

    def generate_visualization(self, df: pd.DataFrame, user_query: str) -> Dict[str, Any]:
        """Generate visualization based on user query."""
        try:
            # Get basic data information
            data_info = self._get_data_info(df)
            
            # Create focused prompt for visualization
            prompt = f"""
            User Query: {user_query}
            Analyze the query and determine the most suitable visualization type (bar, line, scatter, histogram, boxplot, etc.).

            ### DataFrame Information:
            - Columns: {', '.join(data_info['columns'])}
            - Row count: {data_info['row_count']}
            - Numeric columns: {', '.join(data_info['numeric_columns'])}
            - Categorical columns: {', '.join(data_info['categorical_columns'])}
            - Numeric stats: {data_info['numeric_stats']}
            - Category distributions: {data_info['categorical_stats']}

            ### Requirements:
            1. Select the appropriate **visualization type** based on the user query.
            2. Ensure the selected **columns exist** in 'df'.
            3. Handle missing/null values properly (drop, fill, or ignore).
            4. Use **seaborn (`sns`)** for statistical plots and **matplotlib (`plt`)** for basic plots.
            5. Always include:
            - `plt.title()`
            - `plt.xlabel()`, `plt.ylabel()`
            - `plt.legend()` when necessary
            6. The final line of code **must save** the figure using `plt.savefig("output.png")` instead of `plt.show()`.
            7. **DO NOT create example or dummy data.** Your code will run against the actual dataframe provided.
            8. **IMPORTANT: Always use the full dataset for visualization unless explicitly asked to filter.**

            ### Example Code Format (DO NOT COPY, JUST FOLLOW STRUCTURE):
            ```python
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Handle missing values if needed
            df_cleaned = df.dropna(subset=['column_name'])

            plt.figure(figsize=(10,6))
            sns.histplot(df_cleaned["column_name"], kde=True, bins=10)
            plt.title("Descriptive Title")
            plt.xlabel("X Axis Label")
            plt.ylabel("Y Axis Label")
            plt.savefig("output.png")  # Save output to file
            """

            # Generate and validate visualization
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Generate code
                    response = self.model.generate_content(prompt)
                    if not response.candidates or not response.candidates[0].content.parts:
                        raise ValueError("Invalid Gemini response")
                    
                    code = self._extract_python_code(response.candidates[0].content.parts[0].text)
                    if not self._validate_code(code):
                        raise ValueError("Invalid code generated")

                    # Execute and save visualization
                    if self._execute_visualization_code(code, df):
                        visualizations = self._save_figures_to_base64()
                        if not visualizations:
                            raise ValueError("No visualizations generated")
                        
                        return {
                            'status': 'success',
                            'visualizations': visualizations,
                            'code': code
                        }

                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    plt.close('all')
                    
            raise RuntimeError("Failed to generate visualization after multiple attempts")

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'type': str(type(e).__name__)
            }