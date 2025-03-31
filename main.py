# import os
# import logging
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from flask_cors import CORS
# from pinecone import Pinecone
# from datetime import datetime
# from dotenv import load_dotenv
# from flask import Flask, jsonify, request, session
# from werkzeug.utils import secure_filename
# from InsightHelper.InsightHelper import format_insights_for_llm, generate_insight_llm_prompt
# from QueryHelper.QueryHelper import format_context_from_matches, generate_llm_prompt
# from InsightGenerator import DataInsightsGenerator
# from GroqInitialize import initialize_groq_api
# from GeminiHelper.gemini_service import VisualizationService
# from ChatVis.ChatToVisGenerator import ChatToVisGenerator
# from ChatAnalysis.ChatToAnalysisGenerator import ChatToAnalysisGenerator

# # Load environment variables
# load_dotenv()

# # Initialize the Flask app
# app = Flask(__name__)
# CORS(app)

# # Secret key for sessions - CRITICAL for maintaining session state
# app.secret_key = os.getenv("SECRET_KEY", os.urandom(24))

# # Configure session to use filesystem instead of signed cookies (more robust for larger data)
# app.config['SESSION_TYPE'] = 'filesystem'
# app.config['SESSION_PERMANENT'] = True
# app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # 30 minutes

# # Initialize services
# viz_service = VisualizationService()

# # Initialize logging
# logging.basicConfig(level=logging.INFO)

# # Configure upload folder
# UPLOAD_FOLDER = './uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# # Ensure the upload folder exists
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# # Global dictionary to store user data (alternative to session for reliability)
# # Using request IP as key for demonstration; in production use proper authentication
# user_data = {}

# def get_user_id():
#     """Get a unique identifier for the current user"""
#     # In a real application, you'd use a proper user authentication system
#     # For demonstration, we'll use the remote address
#     return request.remote_addr

# def get_user_data():
#     """Get the data for the current user"""
#     user_id = get_user_id()
#     if user_id not in user_data:
#         user_data[user_id] = {
#             "file_info": {
#                 "path": None,
#                 "name": None,
#                 "namespace": None,
#                 "timestamp": None,
#                 "has_embeddings": False,
#                 "has_insights": False
#             },
#             "df_info": {},
#             "insights": None,
#             "visualizations_history": [],
#             "analysis_history": []
#         }
#     return user_data[user_id]

# def save_uploaded_file(file):
#     """Handle file upload and update user data state"""
#     if not file or not file.filename:
#         raise ValueError("No file provided")
    
#     if not file.filename.endswith('.csv'):
#         raise ValueError("Only CSV files are allowed")
    
#     filename = secure_filename(file.filename)
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     file.save(file_path)
    
#     namespace = os.path.splitext(filename)[0]
    
#     user_storage = get_user_data()
#     user_storage['file_info'] = {
#         "path": file_path,
#         "name": filename,
#         "namespace": namespace,
#         "timestamp": datetime.now().isoformat(),
#         "has_embeddings": False,
#         "has_insights": False
#     }
    
#     return file_path

# def create_and_store_dataframe(file_path):
#     """
#     Create DataFrame from uploaded CSV file and store information in user data.
    
#     Args:
#         file_path (str): Path to the uploaded CSV file
        
#     Returns:
#         dict: DataFrame information including shape and columns
#     """
#     try:
#         # Read CSV file into DataFrame
#         df = pd.read_csv(file_path)
        
#         # Store DataFrame metadata in user data
#         df_info = {
#             "rows": len(df),
#             "columns": df.columns.tolist(),
#             "numeric_columns": df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
#             "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
#             "shape": df.shape
#         }
        
#         # Store dataframe info in user data
#         user_storage = get_user_data()
#         user_storage['df_info'] = df_info
        
#         return {
#             "status": "success",
#             "message": "DataFrame created successfully",
#             "df_info": df_info
#         }
        
#     except Exception as e:
#         logging.error(f"Error creating DataFrame: {str(e)}")
#         return {
#             "status": "error",
#             "message": f"Failed to create DataFrame: {str(e)}"
#         }

# def get_dataframe():
#     """
#     Get DataFrame from the file path stored in user data.
    
#     Returns:
#         pandas.DataFrame: DataFrame or None if no file path in user data
#     """
#     user_storage = get_user_data()
#     file_path = user_storage.get('file_info', {}).get('path')
#     if not file_path:
#         return None
    
#     try:
#         return pd.read_csv(file_path)
#     except Exception as e:
#         logging.error(f"Error loading DataFrame: {str(e)}")
#         return None

# # ROUTES:
# # Upload file:
# @app.route('/v1/csv-upload', methods=['POST'])
# def upload_file():
#     """Unified file upload endpoint"""
#     try:
#         if 'file' not in request.files:
#             return jsonify({"status": "fail", "message": "No file part in the request"}), 400
        
#         file = request.files['file']
#         file_path = save_uploaded_file(file)
        
#         # Create and store DataFrame info
#         df_result = create_and_store_dataframe(file_path)
        
#         if df_result["status"] == "error":
#             return jsonify({
#                 "status": "fail",
#                 "message": df_result["message"]
#             }), 500
        
#         user_storage = get_user_data()
#         logging.info(f"Dataframe created successfully: {df_result}")
#         return jsonify({
#             "status": "success",
#             "message": "File uploaded successfully",
#             "file_details": {
#                 "name": user_storage['file_info']["name"],
#                 "timestamp": user_storage['file_info']["timestamp"],
#                 "df_info": df_result["df_info"]
#             }
#         }), 200
        
#     except Exception as e:
#         logging.error(f"Error uploading file: {str(e)}", exc_info=True)
#         return jsonify({"status": "fail", "message": str(e)}), 500

# # Insights:
# @app.route('/v1/csv-insights', methods=['GET'])
# def generate_insights():
#     """Generate insights and visualizations from the uploaded file"""
#     try:
#         user_storage = get_user_data()
#         # Existing file validation
#         file_info = user_storage.get('file_info', {})
#         if not file_info.get("path"):
#             return jsonify({
#                 "status": "fail",
#                 "message": "No file uploaded. Please upload a file first."
#             }), 400

#         # Validate file readability and DataFrame structure
#         try:
#             generator = DataInsightsGenerator(file_info["path"])
#             if generator.df.empty:
#                 return jsonify({
#                     "status": "fail",
#                     "message": "The uploaded file contains no data."
#                 }), 400
#             if len(generator.df.columns) == 0:
#                 return jsonify({
#                     "status": "fail",
#                     "message": "The uploaded file contains no columns."
#                 }), 400
#         except Exception as e:
#             return jsonify({
#                 "status": "fail",
#                 "message": f"Failed to read the uploaded file: {str(e)}"
#             }), 400

#         # Get optional query parameters
#         target_column = request.args.get('target_column')
#         generate_llm_analysis = request.args.get('generate_llm_analysis', 'true').lower() == 'true'

#         logging.info(f"Generating insights for file: {file_info['name']}")
        
#         # Initialize insights dictionary with safe defaults
#         insights = {}
        
#         # Generate each type of insight with individual try-except blocks
#         try:
#             insights["descriptive"] = generator.generate_descriptive_analytics()
#         except Exception as e:
#             logging.warning(f"Failed to generate descriptive analytics: {str(e)}")
#             insights["descriptive"] = {"error": str(e)}

#         try:
#             insights["diagnostic"] = generator.generate_diagnostic_analytics()
#         except Exception as e:
#             logging.warning(f"Failed to generate diagnostic analytics: {str(e)}")
#             insights["diagnostic"] = {"error": str(e)}

#         try:
#             insights["prescriptive"] = generator.generate_prescriptive_analytics()
#         except Exception as e:
#             logging.warning(f"Failed to generate prescriptive analytics: {str(e)}")
#             insights["prescriptive"] = {"error": str(e)}

#         try:
#             insights["outliers"] = generator.detect_outliers()
#         except Exception as e:
#             logging.warning(f"Failed to detect outliers: {str(e)}")
#             insights["outliers"] = {"error": str(e)}

#         # Generate predictive analytics if target column is provided
#         if target_column:
#             if target_column not in generator.df.columns:
#                 insights["predictive"] = {
#                     "error": f"Target column '{target_column}' not found in dataset"
#                 }
#             else:
#                 try:
#                     insights["predictive"] = generator.generate_predictive_analytics(target_column)
#                 except Exception as e:
#                     logging.warning(f"Failed to generate predictive analytics: {str(e)}")
#                     insights["predictive"] = {"error": str(e)}

#         # Convert insights to serializable format
#         try:
#             insights = generator._convert_to_serializable(insights)
#         except Exception as e:
#             logging.error(f"Failed to convert insights to serializable format: {str(e)}")
#             return jsonify({
#                 "status": "fail",
#                 "message": "Failed to process generated insights"
#             }), 500

#         response_data = {
#             "status": "success",
#             "file_details": {
#                 "name": file_info["name"],
#                 "path": file_info["path"]
#             },
#             "insights": insights
#         }

#         # Generate LLM analysis if requested
#         if generate_llm_analysis:
#             logging.info("Generating LLM analysis of insights")
#             try:
#                 insights_context = format_insights_for_llm(insights)
#                 llm_prompt = generate_insight_llm_prompt(insights_context)
#                 llm_response = initialize_groq_api(llm_prompt)
#                 response_data["llm_analysis"] = llm_response
#             except Exception as e:
#                 logging.error(f"Failed to generate LLM analysis: {str(e)}")
#                 response_data["llm_analysis"] = {"error": str(e)}

#         # Update file state in user storage
#         user_storage['file_info']["has_insights"] = True
#         user_storage['insights'] = insights

#         logging.info("Successfully generated insights")
#         return jsonify(response_data), 200

#     except Exception as e:
#         logging.error(f"Error generating insights: {str(e)}", exc_info=True)
#         return jsonify({
#             "status": "fail",
#             "message": str(e),
#             "type": str(type(e).__name__),
#             "details": "An unexpected error occurred while generating insights"
#         }), 500
 
# # Chat to Vis:
# @app.route('/v1/query-visualizations', methods=['POST'])
# def generate_visualizations_from_query():
#     """Generate visualizations based on user query about the uploaded file"""
#     try:
#         user_storage = get_user_data()
#         # Get the user query from request body
#         request_data = request.json
#         if not request_data or 'query' not in request_data:
#             return jsonify({
#                 "status": "fail",
#                 "message": "Missing 'query' in request body"
#             }), 400
            
#         user_query = request_data['query']
        
#         # Validate that file exists and is accessible
#         file_info = user_storage.get('file_info', {})
#         if not file_info or "path" not in file_info:
#             return jsonify({
#                 "status": "fail",
#                 "message": "No file has been uploaded"
#             }), 400

#         # Read and validate DataFrame
#         try:
#             df = pd.read_csv(file_info["path"], encoding='utf-8')
#             if df.empty:
#                 return jsonify({
#                     "status": "fail",
#                     "message": "The uploaded file is empty"
#                 }), 400
#             if len(df.columns) == 0:
#                 return jsonify({
#                     "status": "fail",
#                     "message": "The uploaded file contains no columns"
#                 }), 400
#         except pd.errors.EmptyDataError:
#             return jsonify({
#                 "status": "fail",
#                 "message": "The uploaded file is empty"
#             }), 400
#         except pd.errors.ParserError:
#             return jsonify({
#                 "status": "fail",
#                 "message": "Invalid CSV format"
#             }), 400
#         except Exception as e:
#             return jsonify({
#                 "status": "fail",
#                 "message": f"Failed to read the uploaded file: {str(e)}"
#             }), 400

#         # Create visualization service and generate visualization
#         try:
#             # Initialize the visualization generator
#             viz_generator = ChatToVisGenerator()
            
#             # Generate visualization with enhanced error handling
#             viz_result = viz_generator.generate_visualization(
#                 df=df,
#                 user_query=user_query
#             )
            
#             if viz_result['status'] == 'success':
#                 # Store the visualization in user history
#                 if 'visualizations_history' not in user_storage:
#                     user_storage['visualizations_history'] = []
                    
#                 user_storage['visualizations_history'].append({
#                     'query': user_query,
#                     'timestamp': datetime.now().isoformat(),
#                     'code': viz_result.get('code')
#                 })
                
#                 return jsonify({
#                     "status": "success",
#                     "visualizations": viz_result['visualizations'],
#                     "visualization_code": viz_result['code'],
#                     "query": user_query
#                 }), 200
#             else:
#                 return jsonify({
#                     "status": "fail",
#                     "message": viz_result.get('message', 'Failed to generate visualization'),
#                     "error_type": viz_result.get('type', 'Unknown'),
#                     "query": user_query
#                 }), 400
                
#         except ValueError as ve:
#             return jsonify({
#                 "status": "fail",
#                 "message": str(ve),
#                 "details": "Configuration error"
#             }), 500
#         except Exception as e:
#             logging.error(f"Visualization generation error: {str(e)}", exc_info=True)
#             return jsonify({
#                 "status": "fail", 
#                 "message": "Failed to generate visualization",
#                 "details": str(e)
#             }), 500
            
#     except Exception as e:
#         logging.error(f"Unexpected error in visualization generation: {str(e)}", exc_info=True)
#         return jsonify({
#             "status": "fail",
#             "message": "An unexpected error occurred",
#             "details": str(e)
#         }), 500
              
# # Chat to Analysis:
# @app.route('/v1/query-analysis', methods=['POST'])
# def generate_analysis_from_query():
#     """Generate data analysis based on user query about the uploaded file"""
#     try:
#         user_storage = get_user_data()
#         # Get the user query from request body
#         request_data = request.json
#         if not request_data or 'query' not in request_data:
#             return jsonify({
#                 "status": "fail",
#                 "message": "Missing 'query' in request body"
#             }), 400
            
#         user_query = request_data['query']
            
#         # Validate that file exists and is accessible
#         file_info = user_storage.get('file_info', {})
#         if not file_info or "path" not in file_info:
#             return jsonify({
#                 "status": "fail",
#                 "message": "No file has been uploaded"
#             }), 400

#         # Read and validate DataFrame
#         try:
#             df = pd.read_csv(file_info["path"], encoding='utf-8')
#             if df.empty:
#                 return jsonify({
#                     "status": "fail",
#                     "message": "The uploaded file is empty"
#                 }), 400
#             if len(df.columns) == 0:
#                 return jsonify({
#                     "status": "fail",
#                     "message": "The uploaded file contains no columns"
#                 }), 400
#         except pd.errors.EmptyDataError:
#             return jsonify({
#                 "status": "fail",
#                 "message": "The uploaded file is empty"
#             }), 400
#         except pd.errors.ParserError:
#             return jsonify({
#                 "status": "fail",
#                 "message": "Invalid CSV format"
#             }), 400
#         except Exception as e:
#             return jsonify({
#                 "status": "fail",
#                 "message": f"Failed to read the uploaded file: {str(e)}"
#             }), 400

#         # Create analysis service and generate analysis
#         try:
#             analysis_generator = ChatToAnalysisGenerator()
            
#             analysis_result = analysis_generator.generate_analysis(
#                 df=df,
#                 user_query=user_query
#             )
            
#             if analysis_result['status'] == 'success':
#                 # Recursively convert all non-serializable objects in the result
#                 serializable_result = convert_to_serializable(analysis_result['result'])
                
#                 # Store the analysis in user history
#                 if 'analysis_history' not in user_storage:
#                     user_storage['analysis_history'] = []
                    
#                 user_storage['analysis_history'].append({
#                     'query': user_query,
#                     'timestamp': datetime.now().isoformat(),
#                     'code': analysis_result.get('code')
#                 })
                
#                 return jsonify({
#                     "status": "success",
#                     "result": serializable_result,
#                     "answer": analysis_result['answer'],
#                     "analysis_code": analysis_result['code'],
#                     "query": user_query
#                 }), 200
#             else:
#                 return jsonify({
#                     "status": "fail",
#                     "message": analysis_result.get('message', 'Failed to generate analysis'),
#                     "error_type": analysis_result.get('type', 'AnalysisError'),
#                     "query": user_query
#                 }), 400
                
#         except ValueError as ve:
#             return jsonify({
#                 "status": "fail",
#                 "message": str(ve),
#                 "details": "Configuration error"
#             }), 500
#         except Exception as e:
#             logging.error(f"Analysis generation error: {str(e)}", exc_info=True)
#             return jsonify({
#                 "status": "fail",
#                 "message": "Failed to generate analysis",
#                 "details": str(e)
#             }), 500
            
#     except Exception as e:
#         logging.error(f"Unexpected error in analysis generation: {str(e)}", exc_info=True)
#         return jsonify({
#             "status": "fail",
#             "message": "An unexpected error occurred",
#             "details": str(e)
#         }), 500
      
# def convert_to_serializable(obj):
#     """Recursively convert pandas Series, numpy arrays, and other non-serializable objects to JSON-serializable formats."""
#     if isinstance(obj, (pd.Series, np.ndarray)):
#         return obj.tolist()
#     elif isinstance(obj, dict):
#         return {k: convert_to_serializable(v) for k, v in obj.items()}
#     elif isinstance(obj, (list, tuple)):
#         return [convert_to_serializable(item) for item in obj]
#     else:
#         return obj  

# # Add a user data clear route (helpful for debugging and resetting)
# @app.route('/v1/clear-user-data', methods=['POST'])
# def clear_user_data():
#     user_id = get_user_id()
#     if user_id in user_data:
#         del user_data[user_id]
#     return jsonify({"status": "success", "message": "User data cleared"}), 200

# # Get current user data status (for debugging)
# @app.route('/v1/user-data-status', methods=['GET'])
# def get_user_data_status():
#     user_storage = get_user_data()
#     return jsonify({
#         "status": "success",
#         "has_file": user_storage['file_info'].get('path') is not None,
#         "file_name": user_storage['file_info'].get('name'),
#         "has_insights": user_storage['file_info'].get('has_insights', False),
#         "visualization_history_count": len(user_storage.get('visualizations_history', [])),
#         "analysis_history_count": len(user_storage.get('analysis_history', []))
#     }), 200

# # Run the app
# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)