import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask_cors import CORS
from pinecone import Pinecone
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, jsonify, request, session, send_file
from werkzeug.utils import secure_filename
from flask_session import Session  # Add Flask-Session for server-side sessions
import uuid  # For generating unique session IDs
from InsightHelper.InsightHelper import format_insights_for_llm, generate_insight_llm_prompt
from QueryHelper.QueryHelper import format_context_from_matches, generate_llm_prompt
from InsightGenerator import DataInsightsGenerator
from GroqInitialize import initialize_groq_api
from GeminiHelper.gemini_service import VisualizationService
from ChatVis.ChatToVisGenerator import ChatToVisGenerator
from ChatAnalysis.ChatToAnalysisGenerator import ChatToAnalysisGenerator
from InsightHelper.InsightVis import InsightVisualizer
from ReportGenerartion.ReportGen import InsightReportGenerator
# viz_service = VisualizationService()

load_dotenv()
# Initialize the Flask app
app = Flask(__name__)
CORS(app)

CORS(
    app,
    resources={
        r"/v1/*": {
            "origins": "http://localhost:5173",
            "supports_credentials": True,
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type"]
        }
    }
)

# Set up server-side session configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './session_data'  # Directory to store session files
app.config['SESSION_PERMANENT'] = True  # Make sessions persistent
app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # Session lifetime in seconds (30 minutes)
app.config.update(
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=False  # True in production with HTTPS
)
Session(app)  # Initialize Flask-Session

# Initialize logging  
logging.basicConfig(level=logging.INFO)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure upload folder
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)  # Create session directory

# Session management middleware
@app.before_request
def before_request():
    """Ensure each user has a session_id"""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
        # Initialize user session data
        session['file_data'] = {
            "path": None,
            "name": None,
            "namespace": None,
            "timestamp": None,
            "has_embeddings": False,
            "has_insights": False,
            "has_report": False
        }
        session['namespace_data'] = {
            "name": None,
            "timestamp": None,
            "file_name": None
        }
        session.modified = True
        logging.info(f"Created new session with ID: {session['user_id']}")

def get_user_folder(user_id):
    """Create and return user-specific folder path"""
    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], user_id)
    os.makedirs(user_folder, exist_ok=True)
    return user_folder

def update_current_namespace(namespace, file_name):
    """Update the current active namespace for the user"""
    if 'namespace_data' not in session:
        session['namespace_data'] = {}
    
    session['namespace_data'] = {
        "name": namespace,
        "timestamp": datetime.now().isoformat(),
        "file_name": file_name
    }
    session.modified = True
    logging.info(f"Updated current namespace to: {namespace} from file: {file_name} for user {session['user_id']}")

def save_uploaded_file(file):
    """Handle file upload and update session file state"""
    if not file or not file.filename:
        raise ValueError("No file provided")
    
    if not file.filename.endswith('.csv'):
        raise ValueError("Only CSV files are allowed")
    
    filename = secure_filename(file.filename)
    user_folder = get_user_folder(session['user_id'])
    file_path = os.path.join(user_folder, filename)
    file.save(file_path)
    
    namespace = os.path.splitext(filename)[0]
    
    # Update session file data
    session['file_data'] = {
        "path": file_path,
        "name": filename,
        "namespace": namespace,
        "timestamp": datetime.now().isoformat(),
        "has_embeddings": False,
        "has_insights": False,
        "has_report": False
    }
    session.modified = True
    
    return file_path

def create_and_store_dataframe(file_path):
    """
    Create DataFrame from uploaded CSV file and store dataframe info in session.
    
    Args:
        file_path (str): Path to the uploaded CSV file
        
    Returns:
        dict: DataFrame information including shape and columns
    """
    try:
        # Read CSV file into DataFrame
        df = pd.read_csv(file_path)
        
        # Store the DataFrame path in the session
        session['file_data']['df_path'] = file_path
        session.modified = True
        
        # Get basic DataFrame information
        df_info = {
            "rows": len(df),
            "columns": df.columns.tolist(),
            "numeric_columns": df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object']).columns.tolist()
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
    Get the current DataFrame for the user session.
    
    Returns:
        pandas.DataFrame: Current DataFrame or None if no DataFrame is stored
    """
    if 'file_data' in session and session['file_data'].get('path'):
        try:
            return pd.read_csv(session['file_data']['path'])
        except Exception as e:
            logging.error(f"Error loading DataFrame: {str(e)}")
            return None
    return None

# ROUTES:
# Upload file:
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
        logging.info(f"Dataframe created successfully: {df_result} for user {session['user_id']}")
        
        return jsonify({
            "status": "success",
            "message": "File uploaded successfully",
            "file_details": {
                "name": session['file_data']["name"],
                "timestamp": session['file_data']["timestamp"],
                "df_info": df_result["df_info"]
            }
        }), 200
        
    except Exception as e:
        logging.error(f"Error uploading file: {str(e)}", exc_info=True)
        return jsonify({"status": "fail", "message": str(e)}), 500

# Insights:
@app.route('/v1/csv-insights', methods=['GET'])
def generate_insights():
    """Generate insights and visualizations from the uploaded file"""
    try:
        # Existing file validation
        if 'file_data' not in session or not session['file_data'].get("path"):
            return jsonify({
                "status": "fail",
                "message": "No file uploaded. Please upload a file first."
            }), 400

        # Validate file readability and DataFrame structure
        try:
            file_path = session['file_data']['path']
            generator = DataInsightsGenerator(file_path)
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
        generate_viz = request.args.get('generate_visualizations', 'true').lower() == 'true'

        logging.info(f"Generating insights for file: {session['file_data']['name']} for user {session['user_id']}")
        
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
                "name": session['file_data']["name"],
                "path": session['file_data']["path"]
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

        # Generate visualizations based on insights if requested
        if generate_viz:
            logging.info("Generating insight-based visualizations")
            try:
                dataset_name = session['file_data']["name"]
                
                # Initialize the insight visualizer
                insight_viz = InsightVisualizer()
                
                # Generate visualizations based on insights
                viz_response = insight_viz.generate_insight_visualizations(
                    generator.df, 
                    insights, 
                    dataset_name
                )

                if viz_response['status'] == 'success':
                    response_data["visualizations"] = viz_response['visualizations']
                    response_data["visualization_code"] = viz_response['code']
                else:
                    response_data["visualizations"] = {"error": viz_response['message']}
            except Exception as e:
                logging.error(f"Failed to generate visualizations: {str(e)}")
                response_data["visualizations"] = {"error": str(e)}

        # Update session state
        session['file_data']["has_insights"] = True
        session['file_data']["last_insights"] = insights
        session.modified = True

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
        if 'file_data' not in session or not session['file_data'].get("path"):
            return jsonify({
                "status": "fail",
                "message": "No file has been uploaded"
            }), 400

        # Read and validate DataFrame
        try:
            file_path = session['file_data']['path']
            df = pd.read_csv(file_path, encoding='utf-8')
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
            # Initialize visualization generator
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
@app.route('/v1/query-analysis', methods=['POST'])
def generate_analysis_from_query():
    print("Session data:", session)  # Debug session
    print("Request headers:", request.headers)  # Debug headers
    print("Request JSON:", request.get_json())  # Debug payload
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
        if 'file_data' not in session or not session['file_data'].get("path"):
            return jsonify({
                "status": "fail",
                "message": "No file has been uploaded"
            }), 400

        # Read and validate DataFrame
        try:
            file_path = session['file_data']['path']
            df = pd.read_csv(file_path, encoding='utf-8')
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
                def deep_serialize(obj):
                    if isinstance(obj, (pd.DataFrame, pd.Series)):
                        return obj.to_dict(orient='records')
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.generic):
                        return obj.item()
                    elif isinstance(obj, (int, float, str, bool, type(None))):
                        return obj
                    elif isinstance(obj, dict):
                        return {k: deep_serialize(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [deep_serialize(item) for item in obj]
                    else:
                        return str(obj)

                serialized_result = deep_serialize(analysis_result['result'])
                
                return jsonify({
                    "status": "success",
                    "result": serialized_result,
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

# Add this to your existing Flask app routes
@app.route('/v1/generate-report', methods=['GET'])
def generate_report():
    """Generate a comprehensive report with insights and visualizations"""
    try:
        # Validate file exists
        if 'file_data' not in session or not session['file_data'].get("path"):
            return jsonify({
                "status": "fail",
                "message": "No file uploaded. Please upload a file first."
            }), 400

        # Validate and load file
        try:
            file_path = session['file_data']['path']
            generator = DataInsightsGenerator(file_path)
            df = generator.df  # Get the DataFrame directly
            if df.empty:
                return jsonify({
                    "status": "fail",
                    "message": "The uploaded file contains no data."
                }), 400
            if len(df.columns) == 0:
                return jsonify({
                    "status": "fail",
                    "message": "The uploaded file contains no columns."
                }), 400
        except Exception as e:
            return jsonify({
                "status": "fail",
                "message": f"Failed to read the uploaded file: {str(e)}"
            }), 400

        # Get optional target column parameter
        target_column = request.args.get('target_column')
        
        logging.info(f"Generating report for file: {session['file_data']['name']} for user {session['user_id']}")
        
        # Generate all insights (same as before)
        insights = {}
        insight_types = {
            "descriptive": generator.generate_descriptive_analytics,
            "diagnostic": generator.generate_diagnostic_analytics,
            "prescriptive": generator.generate_prescriptive_analytics,
            "outliers": generator.detect_outliers
        }
        
        for insight_name, insight_func in insight_types.items():
            try:
                insights[insight_name] = insight_func()
            except Exception as e:
                logging.warning(f"Failed to generate {insight_name} analytics: {str(e)}")
                insights[insight_name] = {"error": str(e)}

        # Handle predictive analytics if target column exists
        if target_column and target_column in df.columns:
            try:
                insights["predictive"] = generator.generate_predictive_analytics(target_column)
            except Exception as e:
                logging.warning(f"Failed to generate predictive analytics: {str(e)}")
                insights["predictive"] = {"error": str(e)}

        # Convert insights to serializable format
        try:
            insights = generator._convert_to_serializable(insights)
        except Exception as e:
            logging.error(f"Failed to convert insights: {str(e)}")
            return jsonify({
                "status": "fail",
                "message": "Failed to process generated insights"
            }), 500

        # Generate report with visualizations
        try:
            report_generator = InsightReportGenerator()
            # Generate PDF by default
            report = report_generator.generate_full_report(
                df=df,
                insights=insights,
                dataset_name=session['file_data']["name"],
                target_column=target_column,
                output_format="pdf"
            )

            # Create a response with downloadable PDF
            pdf_path = report["report_pdf_path"]
            
            # Generate a relative URL path to access the PDF
            # This assumes you have a route to serve files from the reports directory
            report_id = report["report_id"]
            pdf_url = f"/api/reports/{report_id}/{session['user_id']}/report.pdf"
            
            # Update session state
            session['file_data']["has_report"] = True
            session['file_data']["last_report"] = report
            session['file_data']["report_pdf_path"] = pdf_path
            session['file_data']["report_pdf_url"] = pdf_url
            session.modified = True

            return jsonify({
                "status": "success",
                "report": {
                    "content": report["report_content"],  # Still include markdown for preview if needed
                    "visualizations": report["visualization_paths"],
                    "insights": insights,
                    "pdf_url": pdf_url  # URL to download the PDF
                },
                "file_details": {
                    "name": session['file_data']["name"],
                    "path": session['file_data']["path"]
                }
            }), 200

        except Exception as e:
            logging.error(f"Report generation failed: {str(e)}")
            return jsonify({
                "status": "fail",
                "message": f"Failed to generate report: {str(e)}"
            }), 500

    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return jsonify({
            "status": "fail",
            "message": "An unexpected error occurred"
        }), 500

@app.route('/api/reports/<report_id>/<user_id>/<filename>', methods=['GET'])
def serve_report_file(report_id, user_id, filename):
    """Serve report files (PDFs, images, etc.) for specific user"""
    # Verify user access (basic security check)
    if user_id != session.get('user_id'):
        return jsonify({
            "status": "fail",
            "message": "Unauthorized access"
        }), 403
    
    report_dir = os.path.join("reports", report_id)
    
    # Check if the file exists
    file_path = os.path.join(report_dir, filename)
    if not os.path.isfile(file_path):
        # Check if it's in visualizations subdirectory
        viz_path = os.path.join(report_dir, "visualizations", filename)
        if os.path.isfile(viz_path):
            file_path = viz_path
        else:
            return jsonify({
                "status": "fail",
                "message": "File not found"
            }), 404
    
    # Determine content type based on file extension
    content_type = "application/pdf" if filename.endswith(".pdf") else "image/png"
    
    # Send file with appropriate headers for download
    response = send_file(
        file_path,
        mimetype=content_type,
        as_attachment=True if filename.endswith(".pdf") else False,
        download_name=filename
    )
    
    # Add headers to make PDF download in browser
    if filename.endswith(".pdf"):
        response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    
    return response

# User Session Management Routes
@app.route('/v1/session/status', methods=['GET'])
def get_session_status():
    """Get current user's session status"""
    return jsonify({
        "status": "success",
        "session_id": session['user_id'],
        "file_data": session.get('file_data', {}),
        "namespace_data": session.get('namespace_data', {})
    }), 200

@app.route('/v1/session/reset', methods=['POST'])
def reset_session():
    """Reset current user's session while preserving user_id"""
    user_id = session.get('user_id')
    session.clear()
    session['user_id'] = user_id
    session['file_data'] = {
        "path": None,
        "name": None,
        "namespace": None,
        "timestamp": None,
        "has_embeddings": False,
        "has_insights": False,
        "has_report": False
    }
    session['namespace_data'] = {
        "name": None,
        "timestamp": None,
        "file_name": None
    }
    session.modified = True
    return jsonify({
        "status": "success",
        "message": "Session reset successfully",
        "session_id": session['user_id']
    }), 200
        
# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)