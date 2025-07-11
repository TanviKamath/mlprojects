import os
import logging
from datetime import datetime

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
if __name__ == "__main__":
    logging.info("Logging setup complete.")
    print(f"Logs will be saved to {LOG_FILE_PATH}")