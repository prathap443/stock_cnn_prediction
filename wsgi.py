"""
WSGI entry point for Gunicorn
"""
from stock_analysis_webapp import app, analyze_all_stocks
import threading
import time
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)  # Fixed: Added __ around name

# Create necessary directories
os.makedirs('data', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Try to load initial data
try:
    analyze_all_stocks()
    logger.info("Initial stock analysis completed")
except Exception as e:
    logger.error(f"Initial analysis error: {str(e)}")

# Define background refresh function
def refresh_data_periodically():
    """Background task to refresh stock data every hour"""
    while True:
        try:
            # Wait before first refresh
            time.sleep(3600)  # 1 hour

            logger.info("Auto-refreshing stock data...")
            analyze_all_stocks()
            logger.info("Auto-refresh complete.")
        except Exception as e:
            logger.error(f"Error in auto-refresh: {str(e)}")

# Start background refresh thread
refresh_thread = threading.Thread(target=refresh_data_periodically, daemon=True)
refresh_thread.start()

# This is what Gunicorn imports
if __name__ == "__main__":  # Fixed: Added __ around name and main
    app.run()