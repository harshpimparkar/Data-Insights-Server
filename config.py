import os

class Config:
    """Base configuration"""
    # Flask settings
    DEBUG = os.getenv('FLASK_DEBUG', 'True') == 'True'
    PORT = int(os.getenv('FLASK_PORT', 5000))
    
    # Upload settings
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', './uploads')
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    
    # API keys
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    # Validate essential configuration
    @classmethod
    def validate(cls):
        """Validate essential configuration is present"""
        if not cls.PINECONE_API_KEY or not cls.PINECONE_INDEX_NAME:
            raise EnvironmentError("Pinecone API key or index name is not set.")