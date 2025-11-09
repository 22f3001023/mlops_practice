import os
from datetime import timedelta
from feast import (
    Entity,
    FeatureView,
    Field,
    FileSource,
    ValueType,
)
from feast.types import Float64, Int64, String


# --- Data Source ---
PROCESSED_DATA_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..",
    "data/processed/stock_data.parquet"
))


processed_stock_data = FileSource(
    path=PROCESSED_DATA_PATH,
    timestamp_field="timestamp",
)


# --- Entity ---
stock_entity = Entity(
    name="stock_id", 
    value_type=ValueType.STRING, 
    description="Stock ticker symbol"
)


# --- Feature View ---
stock_features_view = FeatureView(
    name="stock_features",
    entities=[stock_entity],
    ttl=timedelta(days=7),
    schema=[
        Field(name="rolling_avg_10", dtype=Float64),
        Field(name="volume_sum_10", dtype=Float64),
        # Target is NOT included in the feature schema
    ],
    online=False,
    source=processed_stock_data,
    tags={"project": "stock_predictor"},
)
