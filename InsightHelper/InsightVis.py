from typing import Dict, Any, Optional, List
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
import json

class InsightVisualizer:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the visualization service for insights with Gemini API key."""
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

    def _compress_insights(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Compress insights to reduce context length while preserving key information."""
        compressed = {}

        # Process descriptive insights
        if "descriptive" in insights and isinstance(insights["descriptive"], dict):
            descriptive = {}
            
            # Compress summary stats - keep min, max, mean for numeric columns
            if "summary" in insights["descriptive"]:
                summary = {}
                for col, stats in insights["descriptive"]["summary"].items():
                    if isinstance(stats, dict):
                        # Keep only essential stats
                        compact_stats = {
                            key: stats.get(key) for key in ["min", "max", "mean"] 
                            if key in stats
                        }
                        summary[col] = compact_stats
                descriptive["summary"] = summary
            
            # Keep top N correlations
            if "correlations" in insights["descriptive"]:
                descriptive["correlations"] = insights["descriptive"]["correlations"][:3]
            
            compressed["descriptive"] = descriptive

        # Process diagnostic insights - focus on top patterns and relationships
        if "diagnostic" in insights and isinstance(insights["diagnostic"], dict):
            diagnostic = {}
            
            if "patterns" in insights["diagnostic"]:
                diagnostic["patterns"] = insights["diagnostic"]["patterns"][:3]
                
            if "relationships" in insights["diagnostic"]:
                diagnostic["relationships"] = insights["diagnostic"]["relationships"][:3]
                
            compressed["diagnostic"] = diagnostic

        # Process outlier information - just counts per column
        if "outliers" in insights and isinstance(insights["outliers"], dict) and "detected" in insights["outliers"]:
            outlier_summary = {}
            for col, values in insights["outliers"]["detected"].items():
                if values:
                    outlier_summary[col] = len(values)
            
            compressed["outliers"] = {"summary": outlier_summary}

        # Process predictive analytics - focus on model performance metrics
        if "predictive" in insights and isinstance(insights["predictive"], dict):
            predictive = {}
            
            if "model_performance" in insights["predictive"]:
                predictive["model_performance"] = insights["predictive"]["model_performance"]
                
            if "feature_importance" in insights["predictive"]:
                # Keep only top features
                if isinstance(insights["predictive"]["feature_importance"], dict):
                    features = sorted(
                        insights["predictive"]["feature_importance"].items(),
                        key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
                        reverse=True
                    )
                    predictive["feature_importance"] = dict(features[:5])
                    
            compressed["predictive"] = predictive

        # Process prescriptive insights if available
        if "prescriptive" in insights and isinstance(insights["prescriptive"], dict):
            prescriptive = {}
            
            if "recommendations" in insights["prescriptive"]:
                prescriptive["recommendations"] = insights["prescriptive"]["recommendations"][:3]
                
            compressed["prescriptive"] = prescriptive

        return compressed

    def _extract_visualization_opportunities(self, insights: Dict[str, Any], df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract specific visualization opportunities from insights."""
        opportunities = []
        
        # Check for correlations to visualize
        if "descriptive" in insights and "correlations" in insights["descriptive"]:
            for corr in insights["descriptive"]["correlations"][:3]:
                # Try to extract column names from correlation text
                match = re.search(r'between\s+["\']?([^"\']+)["\']?\s+and\s+["\']?([^"\']+)["\']?', corr)
                if match:
                    col1, col2 = match.groups()
                    if col1 in df.columns and col2 in df.columns:
                        opportunities.append({
                            "type": "correlation",
                            "columns": [col1, col2],
                            "description": corr
                        })
        
        # Check for outliers to visualize
        if "outliers" in insights and "summary" in insights["outliers"]:
            for col, count in insights["outliers"]["summary"].items():
                if col in df.columns and count > 0:
                    opportunities.append({
                        "type": "outliers",
                        "columns": [col],
                        "count": count
                    })
        
        # Check for feature importance to visualize
        if "predictive" in insights and "feature_importance" in insights["predictive"]:
            features = insights["predictive"]["feature_importance"]
            if isinstance(features, dict) and features:
                opportunities.append({
                    "type": "feature_importance",
                    "features": features
                })
        
        # Look for numeric columns with sufficient variance for distribution plots
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols[:3]:  # Limit to first few numeric columns
            if df[col].nunique() > 5:  # Only if there's reasonable variance
                opportunities.append({
                    "type": "distribution",
                    "column": col
                })
        
        # Add time series opportunity if date/time columns exist
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols and len(numeric_cols) > 0:
            opportunities.append({
                "type": "time_series",
                "date_column": date_cols[0],
                "value_column": numeric_cols[0]
            })
            
        return opportunities

    def _format_compressed_insights(self, insights: Dict[str, Any], viz_opportunities: List[Dict[str, Any]], dataset_name: str) -> str:
        """Format compressed insights into a structured prompt for visualization generation."""
        prompt_parts = [f"### Dataset: {dataset_name}"]
        
        # Add each insight category in a compact format
        for category, data in insights.items():
            if isinstance(data, dict):
                prompt_parts.append(f"\n## {category.capitalize()} Insights:")
                
                for key, value in data.items():
                    prompt_parts.append(f"### {key.replace('_', ' ').capitalize()}:")
                    
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            prompt_parts.append(f"- {subkey}: {subvalue}")
                    elif isinstance(value, list):
                        for item in value:
                            prompt_parts.append(f"- {item}")
                    else:
                        prompt_parts.append(f"- {value}")
        
        # Add visualization opportunities
        if viz_opportunities:
            prompt_parts.append("\n## Visualization Opportunities:")
            for i, opp in enumerate(viz_opportunities, 1):
                prompt_parts.append(f"### Opportunity {i}:")
                if opp["type"] == "correlation":
                    prompt_parts.append(f"- Correlation between {opp['columns'][0]} and {opp['columns'][1]}")
                    prompt_parts.append(f"- Description: {opp['description']}")
                elif opp["type"] == "outliers":
                    prompt_parts.append(f"- Outliers in column {opp['columns'][0]}")
                    prompt_parts.append(f"- Count: {opp['count']}")
                elif opp["type"] == "feature_importance":
                    prompt_parts.append("- Feature importance visualization")
                    prompt_parts.append(f"- Top features: {', '.join(list(opp['features'].keys())[:3])}")
                elif opp["type"] == "distribution":
                    prompt_parts.append(f"- Distribution of {opp['column']}")
                elif opp["type"] == "time_series":
                    prompt_parts.append(f"- Time series: {opp['value_column']} over {opp['date_column']}")
        
        return "\n".join(prompt_parts)

    def _get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive DataFrame information for visualization context."""
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Get basic statistics for numeric columns
        stats = {}
        for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
            stats[col] = {
                'mean': round(df[col].mean(), 2),
                'median': round(df[col].median(), 2),
                'min': round(df[col].min(), 2),
                'max': round(df[col].max(), 2)
            }
        
        # Get value counts for categorical columns (limited to top 3 columns, top 3 values each)
        categorical_stats = {}
        for col in df.select_dtypes(exclude=['number']).columns[:3]:
            categorical_stats[col] = {str(k): v for k, v in df[col].value_counts().head(3).to_dict().items()}

        return {
            'columns': df.columns.tolist()[:10],  # Limit to first 10 columns
            'numeric_columns': numeric_cols[:5],  # Limit to first 5 numeric columns
            'categorical_columns': df.select_dtypes(exclude=['number']).columns.tolist()[:5],  # Limit to first 5 categorical
            'row_count': len(df),
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

    def _generate_fallback_visualizations(self, df: pd.DataFrame, num_fallbacks: int = 5) -> str:
        """Generate fallback visualization code that will work regardless of the data."""
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Ensure we have at least some plotting data
        if not numeric_cols:
            # Create a simple count-based visualization if no numeric columns
            code = """
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Fallback 1: Simple bar chart of value counts for first categorical column
plt.figure(figsize=(10, 6))
if df.shape[1] > 0:
    col = df.columns[0]
    counts = df[col].value_counts().head(10)
    plt.bar(counts.index.astype(str), counts.values)
    plt.title(f'Top 10 Values in {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()

# Fallback 2: Count of rows per unique value
plt.figure(figsize=(10, 6))
if df.shape[1] > 0:
    col = df.columns[0]
    df[col].value_counts().head(10).plot(kind='pie', autopct='%1.1f%%')
    plt.title(f'Distribution of {col} Values')
    plt.tight_layout()

# Fallback 3: Heatmap of data presence (not-null values)
plt.figure(figsize=(12, 8))
plt.imshow(~df.head(20).isna(), cmap='viridis', aspect='auto')
plt.colorbar(label='Data Present')
plt.title('Data Presence (First 20 Rows)')
plt.xlabel('Column Index')
plt.ylabel('Row Index')
plt.tight_layout()

# Fallback 4: Count of null vs non-null values by column
plt.figure(figsize=(12, 6))
null_counts = df.isnull().sum()
non_null_counts = df.shape[0] - null_counts
data = pd.DataFrame({'Null': null_counts, 'Not Null': non_null_counts})
data.head(10).plot(kind='barh', stacked=True)
plt.title('Null vs Non-null Values by Column')
plt.tight_layout()

# Fallback 5: Column types visualization
plt.figure(figsize=(10, 6))
type_counts = df.dtypes.value_counts()
plt.pie(type_counts.values, labels=type_counts.index.astype(str), autopct='%1.1f%%')
plt.title('Column Data Types')
plt.tight_layout()
"""
            return code
        
        # If we have numeric columns, create more detailed visualizations
        code = """
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

"""
        # Generate 5 different visualizations based on available columns
        num_numeric = len(numeric_cols)
        num_categorical = len(categorical_cols)
        
        # Viz 1: Distribution of first numeric column
        if num_numeric > 0:
            code += f"""
# Visualization 1: Distribution of {numeric_cols[0]}
plt.figure(figsize=(10, 6))
sns.histplot(df['{numeric_cols[0]}'].dropna(), kde=True)
plt.title('Distribution of {numeric_cols[0]}')
plt.xlabel('{numeric_cols[0]}')
plt.ylabel('Frequency')
plt.tight_layout()
"""
        else:
            code += """
# Visualization 1: Row counts
plt.figure(figsize=(10, 6))
plt.bar(['Total Rows'], [len(df)])
plt.title('Total Number of Rows')
plt.tight_layout()
"""

        # Viz 2: Bar chart of second numeric column
        if num_numeric > 1:
            code += f"""
# Visualization 2: Top values in {numeric_cols[1]}
plt.figure(figsize=(10, 6))
sorted_data = df['{numeric_cols[1]}'].dropna().sort_values(ascending=False).head(10)
plt.bar(range(len(sorted_data)), sorted_data)
plt.title('Top 10 Values in {numeric_cols[1]}')
plt.xlabel('Index')
plt.ylabel('{numeric_cols[1]}')
plt.tight_layout()
"""
        elif num_categorical > 0:
            code += f"""
# Visualization 2: Count of values in {categorical_cols[0]}
plt.figure(figsize=(10, 6))
value_counts = df['{categorical_cols[0]}'].value_counts().head(10)
plt.bar(value_counts.index.astype(str), value_counts.values)
plt.title('Top 10 Values in {categorical_cols[0]}')
plt.xticks(rotation=45)
plt.tight_layout()
"""
        else:
            code += """
# Visualization 2: Column counts
plt.figure(figsize=(10, 6))
plt.bar(['Total Columns'], [df.shape[1]])
plt.title('Total Number of Columns')
plt.tight_layout()
"""

        # Viz 3: Scatter plot of first two numeric columns
        if num_numeric > 1:
            code += f"""
# Visualization 3: Scatter plot of {numeric_cols[0]} vs {numeric_cols[1]}
plt.figure(figsize=(10, 6))
plt.scatter(df['{numeric_cols[0]}'].head(100), df['{numeric_cols[1]}'].head(100), alpha=0.5)
plt.title('{numeric_cols[0]} vs {numeric_cols[1]} (First 100 rows)')
plt.xlabel('{numeric_cols[0]}')
plt.ylabel('{numeric_cols[1]}')
plt.tight_layout()
"""
        elif num_numeric > 0 and num_categorical > 0:
            code += f"""
# Visualization 3: Box plot of {numeric_cols[0]} by {categorical_cols[0]}
plt.figure(figsize=(10, 6))
category_values = df['{categorical_cols[0]}'].value_counts().head(5).index
df_subset = df[df['{categorical_cols[0]}'].isin(category_values)]
sns.boxplot(x='{categorical_cols[0]}', y='{numeric_cols[0]}', data=df_subset)
plt.title('{numeric_cols[0]} by {categorical_cols[0]} Categories')
plt.xticks(rotation=45)
plt.tight_layout()
"""
        else:
            code += """
# Visualization 3: Missing values heatmap
plt.figure(figsize=(10, 6))
plt.imshow(df.head(20).isnull(), cmap='viridis', aspect='auto')
plt.colorbar(label='Missing')
plt.title('Missing Values (First 20 Rows)')
plt.xlabel('Column Index')
plt.ylabel('Row Index')
plt.tight_layout()
"""

        # Viz 4: Line plot or bar chart
        if num_numeric > 0:
            code += f"""
# Visualization 4: Line plot of {numeric_cols[0]} over first 20 rows
plt.figure(figsize=(10, 6))
plt.plot(range(min(20, len(df))), df['{numeric_cols[0]}'].head(20))
plt.title('{numeric_cols[0]} Over First 20 Rows')
plt.xlabel('Row Index')
plt.ylabel('{numeric_cols[0]}')
plt.tight_layout()
"""
        else:
            code += """
# Visualization 4: Column type distribution
plt.figure(figsize=(10, 6))
type_counts = df.dtypes.value_counts()
plt.pie(type_counts.values, labels=type_counts.index.astype(str), autopct='%1.1f%%')
plt.title('Column Data Types')
plt.tight_layout()
"""

        # Viz 5: Correlation heatmap or missing values
        if num_numeric > 1:
            code += """
# Visualization 5: Correlation heatmap
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=['number'])
corr = numeric_df.head(1000).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=.5, cbar_kws={'shrink': .5})
plt.title('Correlation Heatmap of Numeric Columns')
plt.tight_layout()
"""
        else:
            code += """
# Visualization 5: Missing values by column
plt.figure(figsize=(10, 6))
missing = df.isnull().sum()
missing = missing[missing > 0]
if missing.empty:
    plt.text(0.5, 0.5, 'No missing values in the dataset', 
             horizontalalignment='center', verticalalignment='center')
    plt.axis('off')
else:
    missing.sort_values(ascending=False).plot(kind='bar')
    plt.title('Missing Values by Column')
    plt.ylabel('Count')
    plt.tight_layout()
plt.tight_layout()
"""
        
        return code

    def _execute_visualization_code(self, code: str, df: pd.DataFrame) -> bool:
        """Execute the visualization code with enhanced error handling."""
        try:
            # Clear any existing figures
            plt.close('all')
            
            # Execute the code with the dataframe in the namespace
            namespace = {'pd': pd, 'plt': plt, 'sns': sns, 'np': __import__('numpy'), 'df': df}
            exec(code, namespace)
            
            # Verify that at least one figure was created
            if len(plt.get_fignums()) == 0:
                self.logger.warning("No figures were created after code execution")
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Visualization execution error: {str(e)}")
            self.logger.error(traceback.format_exc())
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

    def generate_insight_visualizations(self, df: pd.DataFrame, insights: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
        """Generate visualizations based on compressed data insights with guaranteed output."""
        try:
            # Compress insights to reduce context length
            compressed_insights = self._compress_insights(insights)
            
            # Extract specific visualization opportunities
            viz_opportunities = self._extract_visualization_opportunities(compressed_insights, df)
            
            # Get basic data information (limited scope)
            data_info = self._get_data_info(df)
            
            # Format compressed insights for prompt
            insights_text = self._format_compressed_insights(
                compressed_insights, 
                viz_opportunities,
                dataset_name
            )
            
            # Create focused prompt for visualization
            prompt = f"""
            # Data Visualization Request
            
            You are an expert data scientist tasked with creating insightful, well-labeled, vibrant visualizations based on recommendations provided in this data analysis results text.
            {insights_text}
            
            ### DataFrame Information:
            - Columns: {', '.join(data_info['columns'])}
            - Row count: {data_info['row_count']}
            - Numeric columns: {', '.join(data_info['numeric_columns'])}
            - Categorical columns: {', '.join(data_info['categorical_columns'])}
            
            ### Visualization Tasks:
            1. Create **distinct types of vibrant visualizations** that best depict the insights from the insights text.
            2. Use plt.figure() to start EACH new visualization.
            3. Focus on visualizing the specific recommendations mentioned when possible.
            4. Each visualization should focus on one key element.
            
            ### CRITICAL REQUIREMENTS:
            1. ALWAYS use plt.figure(figsize=(10, 6)) to create each new figure
            2. Create EXACTLY 5 separate visualizations
            3. Do NOT use plt.show() or plt.savefig()
            4. Make sure to create simple but informative visualizations
            5. Only use columns mentioned in the DataFrame Information section
            6. Do not create any example or dummy data - the code will run on the real DataFrame which is already loaded as 'df'
            7. Always use try-except blocks around each visualization to ensure the code doesn't fail
            8. Include plt.tight_layout() at the end of each visualization code
            
            ### Example of ONE good visualization (this is just ONE example, you need to create 5 different ones):
            ```python
            # Visualization 1: Distribution of numeric values
            plt.figure(figsize=(10, 6))
            try:
                sns.histplot(df['{data_info['numeric_columns'][0] if data_info['numeric_columns'] else 'column_name'}'].dropna(), kde=True)
                plt.title('Distribution of {data_info['numeric_columns'][0] if data_info['numeric_columns'] else 'Values'}')
                plt.xlabel('{data_info['numeric_columns'][0] if data_info['numeric_columns'] else 'Values'}')
                plt.ylabel('Frequency')
                plt.tight_layout()
            except Exception as e:
                plt.text(0.5, 0.5, f"Error creating visualization: {{str(e)}}", 
                        horizontalalignment='center', verticalalignment='center')
                plt.axis('off')
                plt.title('Visualization Error')
                plt.tight_layout()
            ```
            
            ### Code Format:
            - Your entire response should be Python code wrapped in ```python and ``` tags
            - Use try-except blocks for EACH visualization to handle errors gracefully
            - The code must create 5 distinct visualizations, each with its own plt.figure() call
            - The code will run directly against 'df' which is already loaded
            """

            # Generate and validate visualization
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Clear any existing plots
                    plt.close('all')
                    
                    # Generate code
                    response = self.model.generate_content(prompt)
                    if not response.candidates or not response.candidates[0].content.parts:
                        raise ValueError("Invalid Gemini response")
                    
                    code = self._extract_python_code(response.candidates[0].content.parts[0].text)
                    if not self._validate_code(code):
                        raise ValueError("Invalid code generated")

                    # Execute the code
                    if self._execute_visualization_code(code, df):
                        visualizations = self._save_figures_to_base64()
                        if visualizations:
                            # Success - we have visualizations
                            return {
                                'status': 'success',
                                'visualizations': visualizations,
                                'code': code
                            }
                
                except Exception as e:
                    self.logger.error(f"Attempt {attempt+1} failed: {str(e)}")
                
                # If we've reached the last attempt or previous attempts failed, use fallback
                if attempt == max_retries - 1:
                    break
            
            # If we get here, use the fallback approach
            self.logger.info("Using fallback visualization generation")
            plt.close('all')
            
            # Generate and execute fallback visualizations
            fallback_code = self._generate_fallback_visualizations(df)
            if self._execute_visualization_code(fallback_code, df):
                visualizations = self._save_figures_to_base64()
                if visualizations:
                    return {
                        'status': 'success_fallback',
                        'message': 'Using fallback visualizations',
                        'visualizations': visualizations,
                        'code': fallback_code
                    }
            
            # If even fallbacks fail, create a single error visualization
            plt.close('all')
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "Could not generate visualizations.\nUsing minimal fallback.", 
                horizontalalignment='center', verticalalignment='center',
                wrap=True, fontsize=14)
            plt.axis('off')
            plt.title('Minimal Fallback Visualization')
            
            visualizations = self._save_figures_to_base64()
            return {
                'status': 'minimal_fallback',
                'message': 'Using minimal fallback visualization',
                'visualizations': visualizations
            }

        except Exception as e:
            self.logger.error(f"Visualization generation error: {str(e)}", exc_info=True)
            
            # Create error visualization as ultimate fallback
            plt.close('all')
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"Error generating visualizations:\n{str(e)}", 
                horizontalalignment='center', verticalalignment='center',
                wrap=True)
            plt.axis('off')
            plt.title('Visualization Error')
            
            visualizations = self._save_figures_to_base64()
            
            return {
                'status': 'error',
                'message': str(e),
                'type': str(type(e).__name__),
                'fallback_visualization': visualizations if visualizations else None
            }