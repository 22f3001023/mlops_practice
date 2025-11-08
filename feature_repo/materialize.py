import pandas as pd
from feast import FeatureStore
from datetime import datetime

def materialize_dynamic():
    """
    Materializes features for the entire time range
    found in the source parquet file.
    """
    print("Loading feature store...")
    store = FeatureStore(repo_path=".")
    
    # 1. Get the source path from the feature view
    view = store.get_feature_view("stock_features")
    source_path = view.source.path
    
    print(f"Reading timestamps from source: {source_path}")
    
    # 2. Read only the timestamp column to find the min/max
    df = pd.read_parquet(source_path, columns=["timestamp"])
    
    # Ensure it's a datetime object
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    start_time = df["timestamp"].min()
    end_time = df["timestamp"].max()
    
    # Add a small buffer to ensure all data is included
    start_time = start_time - pd.Timedelta(seconds=1)
    end_time = end_time + pd.Timedelta(seconds=1)

    print(f"Materializing data from: {start_time} to: {end_time}")

    # 3. Run materialization
    store.materialize(
        feature_views=["stock_features"],
        start_date=start_time,
        end_date=end_time,
    )
    
    print("Materialization complete.")

if __name__ == "__main__":
    materialize_dynamic()
