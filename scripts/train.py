import mlflow
import pandas as pd
import logging
import os
from feast import FeatureStore
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
import dask
from dask.distributed import Client
from multiprocessing import freeze_support


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Dask Configuration (CRITICAL FIX) ---
# Set memory limits and chunking strategy
dask.config.set({
    'dataframe.shuffle.method': 'disk',  # Use disk instead of memory for shuffles
    'array.slicing.split_large_chunks': True,
})


# --- MLflow Configuration ---
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "stock_predictor_v1"
MODEL_NAME = "stock_predictor_rf"


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


# --- Hyperopt Search Space ---
search_space = {
    'n_estimators': scope.int(hp.quniform('n_estimators', 50, 200, 10)),
    'max_depth': scope.int(hp.quniform('max_depth', 3, 15, 1)),
    'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 2, 5, 1)),
}


# --- Global variables for objective function ---
X_train_global, X_test_global, y_train_global, y_test_global = None, None, None, None


def fetch_training_data(store, sample_fraction=0.00001):
    """
    Pulls the entity dataframe and joins features with improved memory handling.
    
    Args:
        store: Feast FeatureStore instance
        sample_fraction: Fraction of data to use (default 0.1 = 10%)
    """
    logging.info("Fetching training data from Feast...")
    
    view = store.get_feature_view("stock_features")
    source_path_actual = view.batch_source.path


    logging.info(f"Reading entity_df (with target) from: {source_path_actual}")
    
    try:
        # Read source parquet in chunks to check size first
        entity_df = pd.read_parquet(
            source_path_actual, 
            columns=["timestamp", "stock_id", "target"]
        )
        logging.info(f"Original entity_df shape: {entity_df.shape}")
    except Exception as e:
        logging.error(f"Failed to read parquet file at {source_path_actual}: {e}")
        return pd.DataFrame()


    # *** FIX 1: Sample data to reduce memory pressure ***
    if sample_fraction < 1.0:
        logging.info(f"Sampling {sample_fraction*100}% of data to reduce memory usage...")
        entity_df = entity_df.sample(frac=sample_fraction, random_state=42)
        logging.info(f"Sampled entity_df shape: {entity_df.shape}")
    
    # *** FIX 2: Rename timestamp to event_timestamp ***
    entity_df = entity_df.rename(columns={"timestamp": "event_timestamp"})
    entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"])
    
    # *** FIX 3: Remove duplicates before join ***
    duplicates = entity_df.duplicated(subset=["stock_id", "event_timestamp"]).sum()
    if duplicates > 0:
        logging.warning(f"Found {duplicates} duplicate rows. Removing...")
        entity_df = entity_df.drop_duplicates(subset=["stock_id", "event_timestamp"])
    
    # *** FIX 4: Aggregate by stock_id and event_timestamp to reduce rows ***
    entity_df = entity_df.groupby(["stock_id", "event_timestamp"]).agg({
        "target": "first"  # Assume target is same for duplicate keys
    }).reset_index()
    
    logging.info(f"Entity DataFrame shape after deduplication: {entity_df.shape}")
    
    # *** FIX 5: Try fetching features with explicit engine handling ***
    try:
        training_data = store.get_historical_features(
            entity_df=entity_df[["event_timestamp", "stock_id"]],
            features=[
                "stock_features:rolling_avg_10",
                "stock_features:volume_sum_10",
            ],
        )
        
        logging.info("Feature retrieval successful. Converting to DataFrame...")
        
        # *** FIX 6: Convert to Pandas with explicit chunking ***
        features_df = training_data.to_df()
        
    except Exception as e:
        logging.error(f"Feature retrieval failed: {e}")
        logging.info("Attempting alternative approach: Direct parquet read and join...")
        
        # Fallback: Read features directly and join
        features_df = pd.read_parquet(source_path_actual)
        features_df = features_df.rename(columns={"timestamp": "event_timestamp"})
        features_df = features_df[["event_timestamp", "stock_id", "rolling_avg_10", "volume_sum_10"]]
        features_df = features_df.drop_duplicates(subset=["stock_id", "event_timestamp"])
    
    logging.info("Feature data fetched successfully.")
    
    # *** FIX 7: Merge with target using inner join to eliminate mismatches ***
    final_df = features_df.merge(
        entity_df[["event_timestamp", "stock_id", "target"]],
        on=["event_timestamp", "stock_id"],
        how="inner"
    )
    
    logging.info(f"Final DataFrame shape: {final_df.shape}")
    
    return final_df


def hyperopt_objective(params):
    """Objective function for Hyperopt, tracked by MLflow."""
    
    with mlflow.start_run(nested=True) as run:
        mlflow.set_tag("model_type", "RandomForestClassifier")
        mlflow.log_params(params)
        
        rf_model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_global, y_train_global)
        
        y_pred = rf_model.predict(X_test_global)
        acc = accuracy_score(y_test_global, y_pred)
        f1 = f1_score(y_test_global, y_pred, average="weighted")
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        return {'loss': -acc, 'status': STATUS_OK, 'run': run, 'model': rf_model}


def register_best_model(trials, parent_run_id):
    """Find the best run and register its model to 'Production'."""
    
    client = mlflow.MlflowClient()
    
    best_trial = trials.best_trial
    best_run_id = best_trial['result']['run'].info.run_id
    best_accuracy = -best_trial['result']['loss']
    logging.info(f"Best run found: {best_run_id} with accuracy: {best_accuracy}")
    
    with mlflow.start_run(run_id=parent_run_id):
        mlflow.sklearn.log_model(
            sk_model=best_trial['result']['model'],
            artifact_path="best_model"
        )


    model_uri = f"runs:/{parent_run_id}/best_model"
    model_details = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    model_version = model_details.version
    
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=model_version,
        stage="Production",
        archive_existing_versions=True
    )
    
    logging.info(f"Registered model version {model_version} to 'Production'.")


if __name__ == "__main__":
    freeze_support()
    
    client = Client(processes=True, memory_limit='4GB', n_workers=4, threads_per_worker=1)
    logging.info(f"Dask dashboard available at: {client.dashboard_link}")

    try:
        store = FeatureStore(repo_path="feature_repo")
    except Exception as e:
        logging.error(f"Failed to load Feast repo at 'feature_repo': {e}")
        exit()


    # *** Adjust sample_fraction based on your data size ***
    # For very large datasets, start with 0.05 (5%)
    # For manageable datasets, use 1.0 (100%)
    df = fetch_training_data(store, sample_fraction=0.1)
    
    if df.empty:
        logging.error("No data fetched from Feast. Exiting.")
        exit()
        
    df = df.dropna()
    
    logging.info(f"Final DataFrame shape after dropna: {df.shape}")
    logging.info(f"Columns: {df.columns.tolist()}")
    
    FEATURES = ["rolling_avg_10", "volume_sum_10"]
    TARGET = "target"
    
    # Validate columns exist
    missing_cols = [col for col in FEATURES + [TARGET] if col not in df.columns]
    if missing_cols:
        logging.error(f"Missing columns in DataFrame: {missing_cols}")
        exit()
    
    X = df[FEATURES]
    y = df[TARGET]
    
    X_train_global, X_test_global, y_train_global, y_test_global = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logging.info(f"Training data shape: {X_train_global.shape}")
    logging.info(f"Test data shape: {X_test_global.shape}")

    logging.info("Starting Hyperopt tuning...")
    trials = Trials()
    with mlflow.start_run(run_name="Hyperopt-Parent-Run") as parent_run:
        best_params = fmin(
            fn=hyperopt_objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=10,
            trials=trials
        )
        
        mlflow.log_params(best_params)
        mlflow.log_metric("best_accuracy", -trials.best_trial['result']['loss'])
    
        logging.info("Registering the best model from the experiment...")
        register_best_model(trials, parent_run.info.run_id)
    
    logging.info("Training pipeline complete.")
