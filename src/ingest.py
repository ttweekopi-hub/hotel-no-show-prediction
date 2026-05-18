import sqlite3
import pandas as pd
import os
from src.logger import get_logger

logger = get_logger("Ingest")

def ingest_data(db_path: str = "data/noshow.db") -> pd.DataFrame:
    """
    Ingests raw data from the SQLite database and returns a Pandas DataFrame.
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
