import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask_cors import CORS
from pinecone import Pinecone
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from InsightHelper.InsightHelper import format_insights_for_llm,generate_insight_llm_prompt
from QueryHelper.QueryHelper import format_context_from_matches,generate_llm_prompt
from InsightGenerator import DataInsightsGenerator
from GroqInitialize import initialize_groq_api
from GeminiHelper.gemini_service import VisualizationService
from ChatVis.ChatToVisGenerator import ChatToVisGenerator
from ChatAnalysis.ChatToAnalysisGenerator import ChatToAnalysisGenerator
# from server.QueryHelper.CSVtoEmbedding import EmbeddingGenerator
# from StoreEmbeddings import store_embeddings
# from QuerytoEmbeddings import query_to_embedding
# chat_to_vis_service = ChatVisualizationService()
# emb_service = EmbeddingGenerator()
# Load environment variables
viz_service = VisualizationService()
load_dotenv()
# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# Get Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
    raise EnvironmentError("Pinecone API key or index name is not set.")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# File
current_dataframe = None
# Dictionary to store namespaces
current_namespace = {
    "name": None,
    "timestamp": None,
    "file_name": None
}

def update_current_namespace(namespace: str, file_name: str):
    """Update the current active namespace"""
    global current_namespace
    current_namespace = {
        "name": namespace,
        "timestamp": datetime.now().isoformat(),
        "file_name": file_name
    }
    logging.info(f"Updated current namespace to: {namespace} from file: {file_name}")

def save_uploaded_file(file):
    """Handle file upload and update current_file state"""
    if not file or not file.filename:
        raise ValueError("No file provided")
    
    if not file.filename.endswith('.csv'):
        raise ValueError("Only CSV files are allowed")
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    namespace = os.path.splitext(filename)[0]
    
    global current_file
    current_file.update({
        "path": file_path,
        "name": filename,
        "namespace": namespace,
        "timestamp": datetime.now().isoformat(),
        "has_embeddings": False,
        "has_insights": False
    })
    
    return file_path

def create_and_store_dataframe(file_path):
    """
    Create DataFrame from uploaded CSV file and store it in memory.
    
    Args:
        file_path (str): Path to the uploaded CSV file
        
    Returns:
        dict: DataFrame information including shape and columns
    """
    try:
        # Read CSV file into DataFrame
        global current_dataframe
        current_dataframe = pd.read_csv(file_path)
        
        # Get basic DataFrame information
        df_info = {
            "rows": len(current_dataframe),
            "columns": current_dataframe.columns.tolist(),
            "numeric_columns": current_dataframe.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            "categorical_columns": current_dataframe.select_dtypes(include=['object']).columns.tolist()
        }
        
        return {
            "status": "success",
            "message": "DataFrame created successfully",
            "df_info": df_info
        }
        
    except Exception as e:
        logging.error(f"Error creating DataFrame: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to create DataFrame: {str(e)}"
        }

def get_current_dataframe():
    """
    Get the currently stored DataFrame.
    
    Returns:
        pandas.DataFrame: Current DataFrame or None if no DataFrame is stored
    """
    return current_dataframe

current_file = {
    "path": None,
    "name": None,
    "namespace": None,
    "timestamp": None,
    "has_embeddings": False,
    "has_insights": False
}

# ROUTES:
@app.route('/v1/csv-upload', methods=['POST'])
def upload_file():
    """Unified file upload endpoint"""
    try:
        if 'file' not in request.files:
            return jsonify({"status": "fail", "message": "No file part in the request"}), 400
        
        file = request.files['file']
        file_path = save_uploaded_file(file)
        
        # Create and store DataFrame
        df_result = create_and_store_dataframe(file_path)
        
        if df_result["status"] == "error":
            return jsonify({
                "status": "fail",
                "message": df_result["message"]
            }), 500
        logging.info(f"Dataframe created successfully: {df_result}")
        return jsonify({
            "status": "success",
            "message": "File uploaded successfully",
            "file_details": {
                "name": current_file["name"],
                "timestamp": current_file["timestamp"],
                "df_info": df_result["df_info"]
            }
        }), 200
        
    except Exception as e:
        logging.error(f"Error uploading file: {str(e)}", exc_info=True)
        return jsonify({"status": "fail", "message": str(e)}), 500

# metadata from the file:
@app.route('/v1/csv-metadata', methods=['GET'])
def get_csv_metadata():
    """Get detailed metadata and sample data from the uploaded CSV file"""
    try:
        if not current_file.get("path"):
            return jsonify({
                "status": "fail", 
                "message": "No file uploaded. Please upload a file first."
            }), 400
            
        file_path = current_file.get("path")
        
        # Get file system stats
        file_stats = os.stat(file_path)
        file_size_bytes = file_stats.st_size
        file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Get the top 5 rows
        sample_data = df.head(5).to_dict(orient='records')
        
        # Get column information
        columns_info = []
        for col in df.columns:
            # Determine the data type
            dtype = str(df[col].dtype)
            python_type = df[col].map(type).value_counts().index[0].__name__
            
            # Get non-null count and percentage
            non_null_count = df[col].count()
            non_null_percentage = round((non_null_count / len(df)) * 100, 2)
            
            # Get unique value count and percentage
            unique_count = df[col].nunique()
            unique_percentage = round((unique_count / len(df)) * 100, 2)
            
            # Get min, max for numeric columns
            min_val = df[col].min() if pd.api.types.is_numeric_dtype(df[col]) else None
            max_val = df[col].max() if pd.api.types.is_numeric_dtype(df[col]) else None
            
            # Get a few sample values
            sample_values = df[col].dropna().unique()[:3].tolist()
            
            columns_info.append({
                "name": col,
                "pandas_dtype": dtype,
                "python_type": python_type,
                "non_null_count": non_null_count,
                "non_null_percentage": non_null_percentage,
                "unique_count": unique_count,
                "unique_percentage": unique_percentage,
                "min_value": min_val,
                "max_value": max_val,
                "sample_values": sample_values
            })
        
        # Build the response
        response = {
            "status": "success",
            "metadata": {
                "filename": current_file.get("name"),
                "file_size_bytes": file_size_bytes,
                "file_size_mb": file_size_mb,
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": columns_info,
                "memory_usage": {
                    "total": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
                    "unit": "MB"
                },
                "has_index": not df.index.equals(pd.RangeIndex(start=0, stop=len(df)))
            },
            "sample_data": sample_data
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logging.error(f"Error generating metadata: {str(e)}", exc_info=True)
        return jsonify({
            "status": "fail",
            "message": str(e),
            "type": str(type(e).__name__),
            "details": "An unexpected error occurred while generating metadata"
        }), 500

@app.route('/v1/csv-insights', methods=['GET'])
def generate_insights():
    """Generate insights and visualizations from the uploaded file"""
    try:
        # Existing file validation
        if not current_file.get("path"):
            return jsonify({
                "status": "fail",
                "message": "No file uploaded. Please upload a file first."
            }), 400

        # Validate file readability and DataFrame structure
        try:
            generator = DataInsightsGenerator(current_file["path"])
            if generator.df.empty:
                return jsonify({
                    "status": "fail",
                    "message": "The uploaded file contains no data."
                }), 400
            if len(generator.df.columns) == 0:
                return jsonify({
                    "status": "fail",
                    "message": "The uploaded file contains no columns."
                }), 400
        except Exception as e:
            return jsonify({
                "status": "fail",
                "message": f"Failed to read the uploaded file: {str(e)}"
            }), 400

        # Get optional query parameters
        target_column = request.args.get('target_column')
        generate_llm_analysis = request.args.get('generate_llm_analysis', 'true').lower() == 'true'
        # generate_viz = request.args.get('generate_visualizations', 'true').lower() == 'true'

        logging.info(f"Generating insights for file: {current_file['name']}")
        
        # Initialize insights dictionary with safe defaults
        insights = {}
        
        # Generate each type of insight with individual try-except blocks
        try:
            insights["descriptive"] = generator.generate_descriptive_analytics()
        except Exception as e:
            logging.warning(f"Failed to generate descriptive analytics: {str(e)}")
            insights["descriptive"] = {"error": str(e)}

        try:
            insights["diagnostic"] = generator.generate_diagnostic_analytics()
        except Exception as e:
            logging.warning(f"Failed to generate diagnostic analytics: {str(e)}")
            insights["diagnostic"] = {"error": str(e)}

        try:
            insights["prescriptive"] = generator.generate_prescriptive_analytics()
        except Exception as e:
            logging.warning(f"Failed to generate prescriptive analytics: {str(e)}")
            insights["prescriptive"] = {"error": str(e)}

        try:
            insights["outliers"] = generator.detect_outliers()
        except Exception as e:
            logging.warning(f"Failed to detect outliers: {str(e)}")
            insights["outliers"] = {"error": str(e)}

        # Generate predictive analytics if target column is provided
        if target_column:
            if target_column not in generator.df.columns:
                insights["predictive"] = {
                    "error": f"Target column '{target_column}' not found in dataset"
                }
            else:
                try:
                    insights["predictive"] = generator.generate_predictive_analytics(target_column)
                except Exception as e:
                    logging.warning(f"Failed to generate predictive analytics: {str(e)}")
                    insights["predictive"] = {"error": str(e)}

        # Convert insights to serializable format
        try:
            insights = generator._convert_to_serializable(insights)
        except Exception as e:
            logging.error(f"Failed to convert insights to serializable format: {str(e)}")
            return jsonify({
                "status": "fail",
                "message": "Failed to process generated insights"
            }), 500

        response_data = {
            "status": "success",
            "file_details": {
                "name": current_file["name"],
                "path": current_file["path"]
            },
            "insights": insights
        }

        # Generate LLM analysis if requested
        if generate_llm_analysis:
            logging.info("Generating LLM analysis of insights")
            try:
                insights_context = format_insights_for_llm(insights)
                llm_prompt = generate_insight_llm_prompt(insights_context)
                llm_response = initialize_groq_api(llm_prompt)
                response_data["llm_analysis"] = llm_response
            except Exception as e:
                logging.error(f"Failed to generate LLM analysis: {str(e)}")
                response_data["llm_analysis"] = {"error": str(e)}

        # # Generate visualizations if requested
        # # Generate visualizations if requested
        # if generate_viz:
        #     logging.info("Generating visualizations")
        #     try:
        #         dataset_name = current_file["name"]  # Extract dataset name
        #         viz_response = viz_service.generate_visualizations(generator.df, insights, dataset_name)

        #         if viz_response['status'] == 'success':
        #             response_data["visualizations"] = viz_response['visualizations']
        #             response_data["visualization_code"] = viz_response['code']
        #         else:
        #             response_data["visualizations"] = {"error": viz_response['message']}
        #     except Exception as e:
        #         logging.error(f"Failed to generate visualizations: {str(e)}")
        #         response_data["visualizations"] = {"error": str(e)}


        # Update file state
        current_file["has_insights"] = True
        current_file["last_insights"] = insights

        logging.info("Successfully generated insights")
        return jsonify(response_data), 200

    except Exception as e:
        logging.error(f"Error generating insights: {str(e)}", exc_info=True)
        return jsonify({
            "status": "fail",
            "message": str(e),
            "type": str(type(e).__name__),
            "details": "An unexpected error occurred while generating insights"
        }), 500
 
# Chat to Vis:
@app.route('/v1/query-visualizations', methods=['POST'])
def generate_visualizations_from_query():
    """Generate visualizations based on user query about the uploaded file"""
    try:
        # Get the user query from request body
        request_data = request.json
        if not request_data or 'query' not in request_data:
            return jsonify({
                "status": "fail",
                "message": "Missing 'query' in request body"
            }), 400
            
        user_query = request_data['query']
        
        # Validate that file exists and is accessible
        if not current_file or "path" not in current_file:
            return jsonify({
                "status": "fail",
                "message": "No file has been uploaded"
            }), 400

        # Read and validate DataFrame
        try:
            df = pd.read_csv(current_file["path"], encoding='utf-8')
            if df.empty:
                return jsonify({
                    "status": "fail",
                    "message": "The uploaded file is empty"
                }), 400
            if len(df.columns) == 0:
                return jsonify({
                    "status": "fail",
                    "message": "The uploaded file contains no columns"
                }), 400
        except pd.errors.EmptyDataError:
            return jsonify({
                "status": "fail",
                "message": "The uploaded file is empty"
            }), 400
        except pd.errors.ParserError:
            return jsonify({
                "status": "fail",
                "message": "Invalid CSV format"
            }), 400
        except Exception as e:
            return jsonify({
                "status": "fail",
                "message": f"Failed to read the uploaded file: {str(e)}"
            }), 400

        # Create visualization service and generate visualization
        try:
            # Initialize with API key from environment
            viz_generator = ChatToVisGenerator()
            
            # Generate visualization with enhanced error handling
            viz_result = viz_generator.generate_visualization(
                df=df,
                user_query=user_query
            )
            
            if viz_result['status'] == 'success':
                return jsonify({
                    "status": "success",
                    "visualizations": viz_result['visualizations'],
                    "visualization_code": viz_result['code'],
                    "query": user_query
                }), 200
            else:
                return jsonify({
                    "status": "fail",
                    "message": viz_result.get('message', 'Failed to generate visualization'),
                    "error_type": viz_result.get('type', 'Unknown'),
                    "query": user_query
                }), 400
                
        except ValueError as ve:
            return jsonify({
                "status": "fail",
                "message": str(ve),
                "details": "Configuration error"
            }), 500
        except Exception as e:
            logging.error(f"Visualization generation error: {str(e)}", exc_info=True)
            return jsonify({
                "status": "fail", 
                "message": "Failed to generate visualization",
                "details": str(e)
            }), 500
            
    except Exception as e:
        logging.error(f"Unexpected error in visualization generation: {str(e)}", exc_info=True)
        return jsonify({
            "status": "fail",
            "message": "An unexpected error occurred",
            "details": str(e)
        }), 500
              
# Chat to Analysis:

def convert_to_serializable(obj):
    """Recursively convert pandas Series, numpy arrays, and other non-serializable objects to JSON-serializable formats."""
    if isinstance(obj, (pd.Series, np.ndarray)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

@app.route('/v1/query-analysis', methods=['POST'])
def generate_analysis_from_query():
    """Generate data analysis based on user query about the uploaded file"""
    try:
        # Get the user query from request body
        request_data = request.json
        if not request_data or 'query' not in request_data:
            return jsonify({
                "status": "fail",
                "message": "Missing 'query' in request body"
            }), 400
            
        user_query = request_data['query']
            
        # Validate that file exists and is accessible
        if not current_file or "path" not in current_file:
            return jsonify({
                "status": "fail",
                "message": "No file has been uploaded"
            }), 400

        # Read and validate DataFrame
        try:
            df = pd.read_csv(current_file["path"], encoding='utf-8')
            if df.empty:
                return jsonify({
                    "status": "fail",
                    "message": "The uploaded file is empty"
                }), 400
            if len(df.columns) == 0:
                return jsonify({
                    "status": "fail",
                    "message": "The uploaded file contains no columns"
                }), 400
        except pd.errors.EmptyDataError:
            return jsonify({
                "status": "fail",
                "message": "The uploaded file is empty"
            }), 400
        except pd.errors.ParserError:
            return jsonify({
                "status": "fail",
                "message": "Invalid CSV format"
            }), 400
        except Exception as e:
            return jsonify({
                "status": "fail",
                "message": f"Failed to read the uploaded file: {str(e)}"
            }), 400

        # Create analysis service and generate analysis
        try:
            analysis_generator = ChatToAnalysisGenerator()
            
            analysis_result = analysis_generator.generate_analysis(
                df=df,
                user_query=user_query
            )
            
            if analysis_result['status'] == 'success':
                # Recursively convert all non-serializable objects in the result
                serializable_result = convert_to_serializable(analysis_result['result'])
                
                return jsonify({
                    "status": "success",
                    "result": serializable_result,
                    "answer": analysis_result['answer'],
                    "analysis_code": analysis_result['code'],
                    "query": user_query
                }), 200
            else:
                return jsonify({
                    "status": "fail",
                    "message": analysis_result.get('message', 'Failed to generate analysis'),
                    "error_type": analysis_result.get('type', 'AnalysisError'),
                    "query": user_query
                }), 400
                
        except ValueError as ve:
            return jsonify({
                "status": "fail",
                "message": str(ve),
                "details": "Configuration error"
            }), 500
        except Exception as e:
            logging.error(f"Analysis generation error: {str(e)}", exc_info=True)
            return jsonify({
                "status": "fail",
                "message": "Failed to generate analysis",
                "details": str(e)
            }), 500
            
    except Exception as e:
        logging.error(f"Unexpected error in analysis generation: {str(e)}", exc_info=True)
        return jsonify({
            "status": "fail",
            "message": "An unexpected error occurred",
            "details": str(e)
        }), 500
        

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)