import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_data(raw_data_path, processed_data_path):
    """
    Reads raw data, performs cleaning and feature engineering, and saves it.
    """
    logging.info(f"Starting data processing from: {raw_data_path}")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)

    # Read the raw dataset
    try:
        df = pd.read_csv(raw_data_path)
        logging.info("Successfully loaded raw data.")
        logging.info(f"Raw data shape: {df.shape}")
    except FileNotFoundError:
        logging.error(f"Error: The file {raw_data_path} was not found.")
        return

    # --- Data Cleaning ---
    # Drop rows with missing values (if any)
    df.dropna(inplace=True)
    logging.info("Performed data cleaning (dropped NA values).")

    # --- Feature Engineering ---
    # Create synthetic features for demonstration
    df['petal_area'] = df['petal_length'] * df['petal_width']
    df['sepal_area'] = df['sepal_length'] * df['sepal_width']
    logging.info("Created new features: 'petal_area' and 'sepal_area'.")

    # --- Add required columns for Feast ---
    # 1. Timestamp Column (synthetic, for demonstration)
    # Create a series of timestamps ending now, for each row
    end_time = datetime.utcnow()
    df['event_timestamp'] = [end_time - timedelta(minutes=i) for i in range(len(df))][::-1]

    # 2. Entity ID Column (assuming each row is a unique observation)
    df['observation_id'] = range(len(df))
    logging.info("Added 'event_timestamp' and 'observation_id' for Feast.")

    # --- Save Processed Data ---
    try:
        df.to_parquet(processed_data_path, index=False)
        logging.info(f"Successfully saved processed data to {processed_data_path}")
        logging.info(f"Processed data shape: {df.shape}")
    except Exception as e:
        logging.error(f"Failed to save processed data: {e}")

if __name__ == '__main__':
    RAW_PATH = 'data/raw/iris.csv'
    PROCESSED_PATH = 'processed_data/stock_data.parquet'
    process_data(RAW_PATH, PROCESSED_PATH)
