from datetime import timedelta
from feast import (
    Entity,
    FeatureView,
    Field,
    FileSource,
    ValueType,
)
from feast.types import Float64, Int64

# --- Data Source ---
# Point Feast to our processed Parquet file.
processed_stock_data = FileSource(
    path="../data/processed/stock_data.parquet",  # Note the ../ to go up one level
    timestamp_field="timestamp",
)

# --- Entity ---
# An entity is the primary key used to look up features.
stock_entity = Entity(
    name="stock_id", 
    value_type=ValueType.STRING, 
    description="Stock ticker symbol"
)

# --- Feature View ---
# Defines the features we want to serve.
stock_features_view = FeatureView(
    name="stock_features",
    entities=[stock_entity],
    ttl=timedelta(days=7),  # How long features are valid (7 days)
    schema=[
        Field(name="rolling_avg_10", dtype=Float64),
        Field(name="volume_sum_10", dtype=Float64),
        Field(name="target", dtype=Int64), # Include target for training data
    ],
    online=True,  # We are only using this for offline training
    source=processed_stock_data,
    tags={"project": "stock_predictor"},
)
