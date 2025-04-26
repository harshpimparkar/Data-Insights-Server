from typing import Dict, Any, Optional, List, Tuple
import logging
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from GeminiHelper.gemini_helper import VisualizationGenerator

class VisualizationService:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key not provided and not found in environment")
            
        self.viz_generator = VisualizationGenerator(self.api_key)
        self.logger = logging.getLogger(__name__)

    def _get_column_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get detailed column metadata for visualization guidance.
        
        Args:
            df (pd.DataFrame): The data to analyze
            
        Returns:
            Dict[str, Any]: Column metadata
        """
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        
        numeric_stats = {}
        for col in numeric_cols:
            numeric_stats[col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean())
            }
            
        categorical_stats = {}
        for col in categorical_cols:
            categorical_stats[col] = {
                'unique_values': df[col].nunique(),
                'top_categories': df[col].value_counts().nlargest(5).index.tolist()
            }
            
        return {
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'numeric_stats': numeric_stats,
            'categorical_stats': categorical_stats,
            'row_count': len(df)
        }

    def _validate_data_for_visualization(self, df: pd.DataFrame, insights: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate data compatibility with relaxed rules for categorical and correlation data.
        
        Args:
            df (pd.DataFrame): The data to visualize
            insights (Dict[str, Any]): The insights dictionary
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        try:
            if df.empty:
                return False, "DataFrame is empty"

            # Get column metadata
            metadata = self._get_column_metadata(df)
            
            # Special handling for different insight types
            for insight_type, data in insights.items():
                if not isinstance(data, dict):
                    continue
                    
                # Skip length validation for certain insight types/keys
                skip_length_validation = [
                    ('diagnostic', 'strong_correlations'),
                    ('diagnostic', 'correlation_matrix'),
                    ('diagnostic', 'category_distributions'),
                    ('descriptive', 'categorical_summaries'),
                    ('outliers', 'categorical_outliers')
                ]
                
                for key, value in data.items():
                    # Skip validation for known exceptions
                    if any(t[0] == insight_type and t[1] == key for t in skip_length_validation):
                        continue
                        
                    # Validate array lengths only for numeric series data
                    if isinstance(value, (list, pd.Series, np.ndarray)):
                        if key.endswith('_values') or key.endswith('_scores'):
                            if len(value) != len(df):
                                return False, f"Length mismatch in {insight_type}.{key}: expected {len(df)}, got {len(value)}"
                
            return True, ""
            
        except Exception as e:
            return False, f"Data validation error: {str(e)}"

    def _create_visualization_prompt(self, insights: Dict[str, Any], df: pd.DataFrame) -> str:
        """
        Create an enhanced prompt that handles mixed data types.
        
        Args:
            insights (Dict[str, Any]): Sanitized insights dictionary
            df (pd.DataFrame): The DataFrame to visualize
            
        Returns:
            str: Enhanced prompt
        """
        metadata = self._get_column_metadata(df)
        
        prompt_parts = [
            "Generate Python visualization code using Matplotlib or Seaborn based on the following data characteristics and insights.",
            "\nData Structure:",
            "\n**IMPORTANT: Always use the full dataset for visualization unless explicitly asked to filter.**",
            f"- Number of rows: {metadata['row_count']}",
            f"- Numeric columns: {', '.join(metadata['numeric_columns'])}",
            f"- Categorical columns: {', '.join(metadata['categorical_columns'])}",
            "\nNumeric Column Ranges:"
        ]
        
        # Add numeric column statistics
        for col, stats in metadata['numeric_stats'].items():
            prompt_parts.append(f"- {col}: range [{stats['min']:.2f} to {stats['max']:.2f}], mean: {stats['mean']:.2f}")
            
        # Add categorical column information
        prompt_parts.append("\nCategorical Column Information:")
        for col, stats in metadata['categorical_stats'].items():
            categories = ', '.join(stats['top_categories'][:3])  # Show top 3 categories
            prompt_parts.append(f"- {col}: {stats['unique_values']} unique values, top categories: {categories}...")

        prompt_parts.append("\nInsights to Visualize:")
        insight_types = {
            'descriptive': 'Descriptive Analytics',
            'diagnostic': 'Diagnostic Analytics',
            'outliers': 'Outliers',
            'predictive': 'Predictive Analytics'
        }

        for key, title in insight_types.items():
            if data := insights.get(key):
                prompt_parts.append(f"\n{title}:\n{data}")

        prompt_parts.append(
            "\nVisualization Requirements:"
            "\n1. Create appropriate visualizations based on data types:"
            "\n   - Use bar plots or pie charts for categorical data"
            "\n   - Use scatter plots, line plots, or histograms for numeric data"
            "\n   - Use heatmaps for correlation matrices"
            "\n2. Handle missing data appropriately"
            "\n3. Set clear titles, labels, and legends"
            "\n4. Use appropriate color schemes for each visualization type"
            "\n5. Ensure all text is readable"
            "\n6. When using Seaborn and specifying a palette, always set hue to the x variable and set legend=False"
        )

        return "\n".join(prompt_parts)

    def generate_visualizations(self, df: pd.DataFrame, insights: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
        """Generate visualizations with improved data type handling and error recovery."""
        try:
            # Preprocess DataFrame and handle categorical data
            df = self.viz_generator._preprocess_dataframe(df)
        
            # Validate that preprocessing worked
            for col in df.columns:
                if isinstance(df[col].dtype, pd.CategoricalDtype):
                    df[col] = df[col].astype(str)
        
            # Create enhanced prompt
            viz_prompt = self._create_visualization_prompt(insights, df)
            self.logger.debug(f"Visualization Prompt:\n{viz_prompt}")
        
            # Generate visualization with retry logic
            max_retries = 3
            last_error = None
            last_code = None
            
            for attempt in range(max_retries):
                try:
                    # Clear any existing plots
                    plt.close('all')
                    
                    success, code, namespace = self.viz_generator.generate_visualization(
                        df=df,
                        insight_text=viz_prompt,
                        dataset_name=dataset_name 
                    )
                    
                    last_code = code  # Save code for debugging
                    
                    if success and self._validate_code(code):
                        self.logger.debug(f"Generated Code:\n{code}")
                        visualization_data = self._save_figures_to_base64()
                    
                        if visualization_data:
                            return {
                                'status': 'success',
                                'visualizations': visualization_data,
                                'code': code
                            }
                        else:
                            # No visualizations but code executed without error
                            # This might happen if the code didn't create any figures
                            last_error = RuntimeError("Code executed but no visualizations were generated")
                    
                except Exception as e:
                    last_error = e
                    self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    plt.close('all')  # Clean up any partial figures
                
                # Modify the prompt for next attempt to avoid the same error
                if attempt < max_retries - 1:
                    viz_prompt += f"\n\nNote: Previous attempt failed. Please use simpler visualization techniques. Avoid using computed values as hue parameters. If using Seaborn, ensure data parameter is properly set."
            
            # After all attempts failed, return a more helpful error
            error_message = f"Failed to generate valid visualization code after {max_retries} attempts: {str(last_error)}"
            self.logger.error(error_message)
            
            return {
                'status': 'error',
                'message': error_message,
                'type': str(type(last_error).__name__) if last_error else "RuntimeError",
                'last_code': last_code  # Include the last attempted code for debugging
            }

        except Exception as e:
            self.logger.error(f"Error in visualization generation: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e),
                'type': str(type(e).__name__)
            }

    def _validate_code(self, code: str) -> bool:
        """Validate the generated Python code for syntax errors."""
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError as e:
            self.logger.error(f"Syntax error in generated code: {str(e)}")
            return False

    def _generate_fallback_visualization(self, df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
        """Generate simple fallback visualizations when AI-generated code fails."""
        visualization_data = {}
        
        try:
            # Close any existing figures
            plt.close('all')
            
            # Create a new figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get numeric columns for a simple visualization
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            if len(numeric_cols) >= 2:
                # Simple scatter plot of first two numeric columns
                df.plot.scatter(
                    x=numeric_cols[0],
                    y=numeric_cols[1],
                    alpha=0.5,
                    title=f"Fallback Visualization: {numeric_cols[0]} vs {numeric_cols[1]}",
                    ax=ax
                )
            elif len(numeric_cols) == 1:
                # Simple histogram of the single numeric column
                df[numeric_cols[0]].plot.hist(
                    title=f"Fallback Visualization: Distribution of {numeric_cols[0]}",
                    ax=ax
                )
            else:
                # Bar chart of the first categorical column's value counts
                cat_col = df.columns[0]
                df[cat_col].value_counts().nlargest(10).plot.bar(
                    title=f"Fallback Visualization: Top 10 values in {cat_col}",
                    ax=ax
                )
            
            # Save the figure to visualization_data
            buf = BytesIO()
            fig.savefig(buf, format='jpg', bbox_inches='tight', dpi=300)
            buf.seek(0)
            visualization_data['visualization_fallback'] = {
                'type': 'image/jpg',
                'data': base64.b64encode(buf.getvalue()).decode('utf-8')
            }
            plt.close(fig)
            buf.close()
            
        except Exception as e:
            self.logger.error(f"Failed to generate fallback visualization: {str(e)}")
        
        return visualization_data
    
    def _save_figures_to_base64(self) -> Dict[str, Dict[str, str]]:
        """Convert all current matplotlib figures to base64 encoded images."""
        visualization_data = {}
        
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            buf = BytesIO()
            
            try:
                fig.savefig(buf, format='jpg', bbox_inches='tight', dpi=300)
                buf.seek(0)
                visualization_data[f'visualization_{fig_num}'] = {
                    'type': 'image/jpg',
                    'data': base64.b64encode(buf.getvalue()).decode('utf-8')
                }
            except Exception as e:
                self.logger.error(f"Failed to save figure {fig_num}: {str(e)}")
            finally:
                plt.close(fig)
                buf.close()
                
        return visualization_data