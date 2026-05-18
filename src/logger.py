import logging
import os
import sys
from datetime import datetime, timezone, timedelta

class SGTFormatter(logging.Formatter):
    """Custom logging formatter for Singapore Standard Time (SGT).

    Locks all log entry timestamps to Singapore Standard Time (SGT, UTC+8)
    regardless of where the pipeline is executed (locally, in a container,
    or on a remote cloud server).
    """
    def formatTime(self, record, datefmt=None):
        """Formats the creation time of the log record.

        Args:
            record: The logging record being processed.
            datefmt: An optional format string for the date/time.

        Returns:
            A string representing the formatted timestamp locked to SGT.
        """
        # Convert epoch timestamp to UTC datetime, then convert to SGT (UTC+8)
        utc_dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        sgt_dt = utc_dt.astimezone(timezone(timedelta(hours=8)))
        if datefmt:
            return sgt_dt.strftime(datefmt)
        return sgt_dt.strftime("%d-%m-%Y %H:%M:%S SGT")

def get_logger(name: str) -> logging.Logger:
    """Configures and returns a timezone-aware logger with SGT timestamps.

    Sets up a logger that streams logs to standard output (stdout) for container
    compatibility and appends them to a centralized log file in logs/pipeline.log.
    Duplicate log propagation is disabled to keep outputs clean.

    Args:
        name: A string representing the name of the logger (e.g., 'Clean', 'Train').

    Returns:
        A configured logging.Logger instance with dual console and file handlers.
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
