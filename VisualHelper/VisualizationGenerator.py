# import logging
# from typing import Dict, List, Optional
# import pandas as pd
# from VisualizationHelper import analyze_dataframe_structure, suggest_visualization_types, generate_code_template
# class VisualizationGenerator:
#     def __init__(self, df: pd.DataFrame, insights: Dict):
#         """
#         Initialize visualization generator with DataFrame and insights.
        
#         Args:
#             df: Input DataFrame
#             insights: Dictionary containing LLM analysis
#         """
#         self.df = df
#         self.insights = insights
#         self.df_analysis = analyze_dataframe_structure(df)
        
#     def generate_visualizations(self, max_plots: int = 3) -> Dict:
#         """
#         Generate visualization code based on data analysis and insights.
        
#         Args:
#             max_plots: Maximum number of plots to generate
            
#         Returns:
#             Dictionary containing generated visualization code and metadata
#         """
#         try:
#             # Get visualization suggestions
#             suggestions = suggest_visualization_types(self.df_analysis, self.insights)
            
#             # Generate code for each suggestion
#             visualizations = []
#             for i, suggestion in enumerate(suggestions[:max_plots]):
#                 code = generate_code_template(suggestion)
#                 if code:
#                     viz_info = {
#                         'id': f'visualization_{i+1}',
#                         'code': code,
#                         'config': suggestion,
#                         'library': suggestion['library'],
#                         'type': suggestion['type']
#                     }
#                     visualizations.append(viz_info)
            
#             return {
#                 'status': 'success',
#                 'visualizations': visualizations,
#                 'total_generated': len(visualizations),
#                 'df_analysis': self.df_analysis
#             }
            
#         except Exception as e:
#             logging.error(f"Error generating visualizations: {str(e)}", exc_info=True)
#             return {
#                 'status': 'error',
#                 'message': str(e),
#                 'type': str(type(e).__name__)
#             }
    
#     def execute_visualization(self, viz_info: Dict) -> Optional[Dict]:
#         """
#         Execute generated visualization code and return the result.
        
#         Args:
#             viz_info: Visualization information dictionary
            
#         Returns:
#             Dictionary containing the executed visualization result
#         """
#         try:
#             # Create local namespace for execution
#             namespace = {'df': self.df}
            
#             # Execute the code
#             exec(viz_info['code'], namespace)
            
#             # Get the visualization function
#             func_name = f"create_{viz_info['type']}_plot"
#             if func_name in namespace:
#                 result = namespace[func_name](self.df)
#                 return {
#                     'status': 'success',
#                     'visualization': result,
#                     'type': viz_info['type']
#                 }
            
#             return None
            
#         except Exception as e:
#             logging.error(f"Error executing visualization: {str(e)}", exc_info=True)
#             return {
#                 'status': 'error',
#                 'message': str(e),
#                 'type': str(type(e).__name__)
#             }