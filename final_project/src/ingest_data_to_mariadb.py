import pandas as pd
import mysql.connector
import os
import logging
import argparse
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MariaDB connection details
DB_HOST = "127.0.0.1" # Docker container is mapped to localhost
DB_USER = "root"
DB_PASSWORD = "admin@123"
DB_NAME = "airflow_db"

def get_sql_type(dtype):
    if pd.api.types.is_integer_dtype(dtype):
        return "BIGINT"
    elif pd.api.types.is_float_dtype(dtype):
        return "FLOAT"
    elif pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    else:
        return "TEXT"

def ingest_data(data_file_path, table_name):
    try:
        # 1. Read data from CSV
        if not os.path.exists(data_file_path):
            logger.error(f"Data file not found at: {data_file_path}")
            raise FileNotFoundError(f"Data file not found at {data_file_path}")
        
        df = pd.read_csv(data_file_path)
        df = df.replace({np.nan: None})
        logger.info(f"Successfully read data from {data_file_path}. Shape: {df.shape}")

        # 2. Connect to MariaDB
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = conn.cursor()
        logger.info("Connected to MariaDB.")

        # Create database if not exists
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
        conn.database = DB_NAME
        logger.info(f"Database '{DB_NAME}' ensured to exist.")

        # Drop table if it exists
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        logger.info(f"Dropped table '{table_name}' if it existed.")

        # Create table
        columns_with_types = ", ".join([f"`{col}` {get_sql_type(df[col].dtype)}" for col in df.columns])
        create_table_query = f"CREATE TABLE {table_name} ({columns_with_types})"
        cursor.execute(create_table_query)
        logger.info(f"Table '{table_name}' created with schema: {columns_with_types}")

        # 3. Ingest data
        # Prepare insert statement
        placeholders = ", ".join(["%s"] * len(df.columns))
        insert_query = f"INSERT INTO {table_name} VALUES ({placeholders})"

        # Convert DataFrame to list of tuples for insertion
        data_to_insert = [tuple(row) for row in df.values]
        cursor.executemany(insert_query, data_to_insert)
        conn.commit()
        logger.info(f"Successfully ingested {len(data_to_insert)} rows into '{table_name}'.")

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        raise
    except mysql.connector.Error as err:
        logger.error(f"MariaDB error: {err}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()
            logger.info("MariaDB connection closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ingest data from a CSV file to a MariaDB table.')
    parser.add_argument('--file_path', required=True, help='Path to the CSV file to ingest.')
    parser.add_argument('--table_name', required=True, help='Name of the table to ingest data into.')
    args = parser.parse_args()
    
    ingest_data(args.file_path, args.table_name)