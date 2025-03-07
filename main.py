import os
import logging
from flask import Flask
from flask_cors import CORS
from pinecone import Pinecone
from dotenv import load_dotenv
from config import Config
from Routes import register_routes

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)

def create_app(config_class=Config):
    """Application factory function"""
    # Initialize the Flask app
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Configure CORS
    CORS(app)
    
    # Ensure the upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Initialize Pinecone client
    pc = Pinecone(api_key=app.config['PINECONE_API_KEY'])
    index = pc.Index(app.config['PINECONE_INDEX_NAME'])
    
    # Store important objects in app.config for access across routes
    app.config['PINECONE_INDEX'] = index
    
    # Register all routes
    register_routes(app)
    
    return app

# Create the application instance
app = create_app()

# Run the app
if __name__ == '__main__':
    app.run(debug=app.config['DEBUG'], host='0.0.0.0', port=app.config['PORT'])