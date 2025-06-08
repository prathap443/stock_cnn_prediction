"""
WSGI entry point for Gunicorn
"""
from app import app, analyze_all_stocks
import threading
import time
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs('data', exist_ok=True)
os.makedirs('templates', exist_ok=True)

def initial_load():
    """Run initial stock analysis after app starts."""
    try:
        logger.info("Running initial stock analysis in background...")
        analyze_all_stocks()
        logger.info("Initial stock analysis completed")
    except Exception as e:
        logger.error(f"Initial analysis error: {str(e)}")

def refresh_data_periodically():
    """Background task to refresh stock data every hour"""
    while True:
        try:
            time.sleep(3600)
            logger.info("Auto-refreshing stock data...")
            analyze_all_stocks()
            logger.info("Auto-refresh complete.")
        except Exception as e:
            logger.error(f"Error in auto-refresh: {str(e)}")

# Start both threads after short delay (to let app load first)
threading.Timer(5, initial_load).start()
threading.Thread(target=refresh_data_periodically, daemon=True).start()

# Gunicorn expects 'app' here
app = app
