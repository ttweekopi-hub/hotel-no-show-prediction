"""
This module handles raw data ingestion from our SQLite database.

I built this module to safely establish a connection with the local SQLite database file,
run a SELECT query to retrieve all rows from the 'noshow' table, load them into a structured
Pandas DataFrame, and cleanly close the database connection afterward.
"""

import sqlite3
import pandas as pd
import os
from src.logger import get_logger

logger = get_logger("Ingest")

def ingest_data(db_path: str = "data/noshow.db") -> pd.DataFrame:
    """Ingests raw hotel booking records from the SQLite database.

    Establishes a connection to the SQLite database, pulls all rows from the
    'noshow' table, and returns them as a structured Pandas DataFrame.

    Args:
        db_path: The file path to the SQLite database. Defaults to 'data/noshow.db'.

    Returns:
        A pd.DataFrame containing the raw records loaded from the database.

    Raises:
        FileNotFoundError: If the database file does not exist at db_path.
    """
    if not os.path.exists(db_path):
        err_msg = f"Database not found at path: {db_path}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)
        
    logger.info(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    
    try:
        query = "SELECT * FROM noshow"
        logger.info("Executing query: SELECT * FROM noshow")
        df = pd.read_sql_query(query, conn)
        logger.info(f"Successfully loaded {len(df)} rows from the database.")
        return df
    except Exception as e:
        logger.error(f"Error executing database query: {str(e)}")
        raise e
    finally:
        conn.close()
        logger.info("Database connection closed.")
