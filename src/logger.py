import logging
import os
import sys
from datetime import datetime, timezone, timedelta

class SGTFormatter(logging.Formatter):
    """
    Custom logging formatter that locks timestamps to Singapore Standard Time (SGT, UTC+8)
    and formats them as DD-MM-YYYY HH:MM:SS SGT.
    """
    def formatTime(self, record, datefmt=None):
        # Convert epoch timestamp to UTC datetime, then convert to SGT (UTC+8)
        utc_dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        sgt_dt = utc_dt.astimezone(timezone(timedelta(hours=8)))
        if datefmt:
            return sgt_dt.strftime(datefmt)
        return sgt_dt.strftime("%d-%m-%Y %H:%M:%S SGT")

def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured SGT timezone-aware logger with dual output (Console & File).
    Guarantees duplicate log prevention and Docker/WSL/terminal compatibility.
    """
    logger = logging.getLogger(name)
    
    # If the logger is already configured, return it to prevent duplicate handlers
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Disable propagation to prevent duplicate logs in parent handlers
    
    # Ensure centralized logs folder exists inside workspace
    os.makedirs("logs", exist_ok=True)
    
    # SGT Custom Formatter
    formatter = SGTFormatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt="%d-%m-%Y %H:%M:%S SGT"
    )
    
    # Console Handler: Uses sys.stdout explicitly for reliable streaming in Docker/WSL
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler: Appends logs to centralized file
    file_handler = logging.FileHandler("logs/pipeline.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
