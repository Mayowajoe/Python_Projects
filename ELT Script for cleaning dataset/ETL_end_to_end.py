# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 21:29:18 2025

@author: adebo
"""

import pandas as pd
import numpy as np
import sqlite3
import logging

# Set up logging for better feedback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_data(file_path: str, header: int = 0, encoding: str = 'utf-8', sep: str = ',') -> pd.DataFrame:
    """
    Extract: Load a CSV file into a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path, header=header, encoding=encoding, sep=sep)
        logging.info(f"Successfully extracted data from '{file_path}'.")
        return df
    except FileNotFoundError:
        logging.error(f"Error: The file '{file_path}' was not found.")
        raise
    except Exception as e:
        logging.error(f"An error occurred during data extraction: {e}")
        raise

def infer_and_convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infers and converts data types for a DataFrame.
    """
    for col in df.columns:
        # Attempt to convert to numeric, coercing errors
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception:
            pass  # Not a numeric column

    # Convert to best possible types, including string and datetime
    df = df.convert_dtypes()

    # Attempt to parse any column with datetime-like values
    for col in df.columns:
        # Skip numeric columns to avoid issues
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        try:
            # Check if parsing to datetime is a good idea (has some non-null values)
            if df[col].astype(str).str.contains(r'\d{4}-\d{2}-\d{2}', na=False).any():
                df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
        except Exception as e:
            logging.warning(f"Could not convert column '{col}' to datetime: {e}")
            pass
    
    return df

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform: Clean and standardize the DataFrame.
    """
    if df.empty:
        logging.warning("Input DataFrame is empty. Transformation skipped.")
        return df

    # Drop rows and columns where all values are null
    df = df.dropna(axis=0, how='all')
    df = df.dropna(axis=1, how='all')

    # Drop rows with more than 50% missing values
    thresh = len(df.columns) // 2
    df = df.dropna(thresh=thresh, axis=0)
    
    # Fill remaining nulls
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        else:
            mode_vals = df[col].mode()
            if not mode_vals.empty:
                df[col] = df[col].fillna(mode_vals[0])
            else:
                df[col] = df[col].fillna('')

    # Standardize column names
    df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]

    # Infer and convert data types
    df = infer_and_convert_types(df)

    # Add optional metadata columns
    df.insert(0, 'row_id', range(1, len(df) + 1))
    df['null_count'] = df.isnull().sum(axis=1)

    logging.info("Data transformation completed.")
    return df

def load_data_to_csv(df: pd.DataFrame, output_path: str):
    """
    Load: Save the cleaned DataFrame to a CSV file.
    """
    try:
        df.to_csv(output_path, index=False)
        logging.info(f"Data successfully loaded to CSV file: '{output_path}'.")
    except Exception as e:
        logging.error(f"An error occurred during CSV loading: {e}")
        raise

def load_data_to_sqlite(df: pd.DataFrame, db_path: str, table_name: str):
    """
    Load (optional): Save the DataFrame to a SQLite table.
    """
    try:
        conn = sqlite3.connect(db_path)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
        logging.info(f"Data successfully loaded to SQLite table '{table_name}' in '{db_path}'.")
    except Exception as e:
        logging.error(f"An error occurred during SQLite loading: {e}")
        raise

def main():
    """
    Orchestrates the ETL pipeline.
    """
    logging.info("Starting the ETL process.")

    # --- Configuration ---
    input_csv = 'input_data.csv'
    output_csv = 'cleaned_data.csv'
    sqlite_db = 'data.db'
    table_name = 'my_table'

    # --- Step 1: Extract ---
    raw_data = extract_data(input_csv)
    if raw_data is None or raw_data.empty:
        logging.warning("No data to process. Exiting.")
        return

    # --- Step 2: Transform ---
    cleaned_data = transform_data(raw_data)

    # --- Step 3: Load ---
    # Load to CSV
    load_data_to_csv(cleaned_data, output_csv)
    
    # (Optional) Load to SQLite
    # load_data_to_sqlite(cleaned_data, sqlite_db, table_name)

    logging.info("ETL process finished.")

if __name__ == '__main__':
    main()