import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directories():
    """
    Create all required directories for the application.
    """
    dirs = [
        "data",
        "models/weights",
        "utils",
        ".streamlit"
    ]
    
    for d in dirs:
        os.makedirs(os.path.join(os.path.dirname(__file__), d), exist_ok=True)
        logger.info(f"Directory '{d}' created or already exists.")

def setup():
    """
    Run all setup tasks.
    """
    logger.info("Starting setup...")
    create_directories()
    logger.info("Setup completed successfully.")

if __name__ == "__main__":
    setup()