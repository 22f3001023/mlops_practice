import pandas as pd
import glob
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(data_paths):
    """Loads and concatenates all CSV files from a list of directories."""
    all_files = []
    for path in data_paths:
        all_files.extend(glob.glob(os.path.join(path, "*.csv")))
    
    if not all_files:
        logging.warning("No CSV files found in specified paths.")
        return pd.DataFrame()

    logging.info(f"Found {len(all_files)} files to process.")
    
    df_list = []
    for f in all_files:
        try:
            # Extract stock ID from filename
            stock_id = os.path.basename(f).split('_')[0]
            
            df = pd.read_csv(f)
            df['stock_id'] = stock_id
            df_list.append(df)
            logging.info(f"Loaded {f} for stock {stock_id}")
        except Exception as e:
            logging.error(f"Error loading {f}: {e}")
            
    return pd.concat(df_list, ignore_index=True)

def fill_missing_timestamps(df):
    """Fills missing 1-minute gaps in time-series data per stock."""
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # IMPORTANT: Sort by stock and time before doing anything else
    df = df.sort_values(by=['stock_id', 'timestamp'])
    
    # Create a complete time index for each stock
    df_full = df.set_index('timestamp').groupby('stock_id').apply(lambda x: x.asfreq('1min'), include_groups=False)
    
    # Forward-fill the missing values
    df_full = df_full.ffill()
    
    # Reset index to bring back 'stock_id' and 'timestamp'
    df_full = df_full.reset_index()
    logging.info("Filled missing timestamps using forward-fill (ffill).")
    return df_full

def create_features_and_target(df):
    """Engineers features and the target variable."""
    
    # Group by stock_id to ensure features are calculated per stock
    grouped = df.groupby('stock_id')
    
    # 1. Feature: rolling_avg_10 (10-min moving average of close price)
    # Includes current minute 't' and 9 previous (t-9 to t)
    df['rolling_avg_10'] = grouped['close'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    
    # 2. Feature: volume_sum_10 (Total volume traded over 10 min)
    # Includes current minute 't' and 10 previous (t-10 to t)? -> Problem statement says "t-10 to t" (11 mins) but also "10 min".
    # Let's assume (t-9 to t) for a 10-minute window, consistent with the rolling average.
    df['volume_sum_10'] = grouped['volume'].transform(lambda x: x.rolling(window=10, min_periods=1).sum())
    
    # 3. Target: Predict 5 mins ahead
    # We need to look at the price 5 minutes from now.
    df['target'] = grouped['close'].transform(lambda x: x.shift(-5))
    
    # Create binary target: 1 if future price is higher, 0 otherwise
    df['target'] = (df['target'] > df['close']).astype(int)
    
    logging.info("Created features (rolling_avg_10, volume_sum_10) and target.")
    return df

def main():
    # Define data directories for all iterations
    # This script will process ALL data it finds
    RAW_DATA_PATHS = ['data/raw/v0', 'data/raw/v1']
    PROCESSED_DATA_DIR = 'data/processed'
    OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, 'stock_data.parquet')
    
    # Create output directory if it doesn't exist
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    logging.info("Starting data processing pipeline...")
    
    # 1. Load data
    data = load_data(RAW_DATA_PATHS)
    if data.empty:
        logging.error("No data loaded. Exiting.")
        return
        
    # 2. Fill missing timestamps
    data_filled = fill_missing_timestamps(data)
    
    # 3. Create features and target
    data_features = create_features_and_target(data_filled)
    
    # 4. Clean up
    # Drop rows with NaN values (e.g., the last 5 rows where target can't be computed)
    final_data = data_features.dropna()
    
    # 5. Save processed data
    final_data.to_parquet(OUTPUT_FILE, index=False)
    logging.info(f"Successfully processed data and saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
