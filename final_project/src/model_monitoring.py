import os
import pandas as pd
import logging
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.pipeline.column_mapping import ColumnMapping
import requests # NEW IMPORT

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI endpoint for predictions
FASTAPI_PREDICT_URL = "http://fastapi_app:8000/predict" # Use service name for Docker internal network

def run_model_monitoring():
    """
    Runs Evidently reports for data drift and regression model performance.
    """
    project_root = '/opt/airflow'
    processed_data_path = os.path.join(project_root, 'data', 'processed', 'manga_processed.parquet')
    monitoring_reports_dir = os.path.join(project_root, 'data', 'monitoring_reports')
    
    os.makedirs(monitoring_reports_dir, exist_ok=True)

    logger.info(f"Loading processed data from {processed_data_path} for monitoring.")
    current_data = pd.read_parquet(processed_data_path)
    logger.info(f"Processed data loaded. Shape: {current_data.shape}")

    # Fill missing values in relevant columns for Evidently
    for col in ['popularity', 'rank_val', 'volumes', 'chapters', 'primary_author', 'primary_genre', 'primary_demographic', 'primary_serialization']:
        if col in current_data.columns:
            current_data[col] = current_data[col].fillna(0)

    # Ensure 'score' column is numeric for Evidently
    current_data['score'] = pd.to_numeric(current_data['score'], errors='coerce')
    # Drop rows with NaN in 'score' for the entire dataset before splitting
    current_data.dropna(subset=['score'], inplace=True)

    # For demonstration, we'll use a subset of the current data as reference data.
    # In a real-world scenario, reference data would be your training dataset.
    # And current data would be recent production data.
    reference_data = current_data.sample(frac=0.5, random_state=42)
    production_data = current_data.drop(reference_data.index)

    reference_data.reset_index(drop=True, inplace=True)
    production_data.reset_index(drop=True, inplace=True)

    logger.info(f"NaNs in production_data['score'] before prediction: {production_data['score'].isnull().sum()}")
    logger.info(f"Shape of production_data before prediction: {production_data.shape}")

    # Drop 'images' column as it's empty and causes issues with Evidently
    if 'images' in reference_data.columns:
        reference_data.drop(columns=['images'], inplace=True)
    if 'images' in production_data.columns:
        production_data.drop(columns=['images'], inplace=True)

    # Data Drift Report
    data_drift_report = Report(metrics=[DataDriftPreset()])
    column_mapping_obj = ColumnMapping()
    logger.info("Running data drift report...")
    data_drift_report.run(reference_data=reference_data, current_data=production_data, column_mapping=column_mapping_obj)
    logger.info("Data drift report run complete.")
    data_drift_report_path = os.path.join(monitoring_reports_dir, 'data_drift_report.html')
    data_drift_report.save_html(data_drift_report_path)
    logger.info(f"Data Drift Report saved to {data_drift_report_path}")

    # Regression Model Performance Report
    # Fetch predictions from FastAPI app
    # logger.info("Fetching predictions from FastAPI app...")
    # predictions = []
    # # Define the expected features for the FastAPI model based on MangaFeatures Pydantic model
    # # This should match src/app.py -> MangaFeatures
    # fastapi_features = [
    #     "manga_info_id", "mal_id", "publishing", "approved", 
    #     "scored_by", "members", "favorites", "volumes", "chapters"
    # ]

    # # Ensure production_data has all required FastAPI features, fill missing with 0 if necessary
    # for col in fastapi_features:
    #     if col not in production_data.columns:
    #         production_data[col] = 0 # Add missing columns with default 0
    #     production_data[col] = pd.to_numeric(production_data[col], errors='coerce').fillna(0) # Ensure numeric and no NaNs

    # for index, row in production_data.iterrows():
    #     try:
    #         # Create payload matching MangaFeatures Pydantic model
    #         payload = {
    #             "manga_info_id": int(row["manga_info_id"]),
    #             "mal_id": int(row["mal_id"]),
    #             "publishing": bool(row["publishing"]),
    #             "approved": bool(row["approved"]),
    #             "scored_by": int(row["scored_by"]),
    #             "members": int(row["members"]),
    #             "favorites": int(row["favorites"]),
    #             "volumes": int(row["volumes"]),
    #             "chapters": int(row["chapters"])
    #         }
    #         response = requests.post(FASTAPI_PREDICT_URL, json=payload)
    #         response.raise_for_status() # Raise an exception for HTTP errors

    #         response_json = response.json()
    #         predicted_score_raw = response_json.get("predicted_score")

    #         if predicted_score_raw is None or (isinstance(predicted_score_raw, (int, float)) and pd.isna(predicted_score_raw)):
    #             logger.warning(f"FastAPI returned None or NaN for predicted_score for row {index}. Appending 0.0.")
    #             predicted_score = 0.0
    #         else:
    #             predicted_score = float(predicted_score_raw) # Ensure float
    #         predictions.append(predicted_score)
    #     except requests.exceptions.RequestException as e:
    #         logger.error(f"HTTP Request error fetching prediction for row {index}: {e}. Status: {response.status_code if 'response' in locals() else 'N/A'}, Response: {response.text if 'response' in locals() else 'N/A'}")
    #         predictions.append(0.0) # Append a default value on error
    #     except Exception as e:
    #         logger.error(f"General error fetching prediction for row {index}: {e}. Response JSON: {response.text if 'response' in locals() else 'N/A'}")
    #         predictions.append(0.0) # Append a default value on error
    
    # if len(predictions) != len(production_data):
    #     error_msg = f"Mismatch in prediction count. Expected {len(production_data)}, got {len(predictions)}"
    #     logger.error(error_msg)
    #     raise ValueError(error_msg)

    # production_data['prediction'] = predictions
    # production_data['prediction'].fillna(0, inplace=True) # Final check for any NaNs

    # regression_performance_report = Report(metrics=[RegressionPreset()])
    # regression_column_mapping = ColumnMapping(target='score', prediction='prediction')
    # regression_performance_report.run(reference_data=reference_data, current_data=production_data, 
    #                                   column_mapping=regression_column_mapping)
    # regression_performance_report_path = os.path.join(monitoring_reports_dir, 'regression_performance_report.html')
    # regression_performance_report.save_html(regression_performance_report_path)
    # logger.info(f"Regression Model Performance Report saved to {regression_performance_report_path}")

if __name__ == '__main__':
    run_model_monitoring()