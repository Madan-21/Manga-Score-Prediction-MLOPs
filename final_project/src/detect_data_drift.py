import pandas as pd
from sqlalchemy import create_engine
import logging
from urllib.parse import quote_plus
from scipy.stats import ks_2samp, chi2_contingency
import mlflow
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MariaDB connection details
DB_HOST = "127.0.0.1"
DB_USER = "root"
DB_PASSWORD = "admin@123"
DB_NAME = "airflow_db"
TABLE_NAME = "manga_data"
DRIFT_REPORT_DIR = "drift_reports"

def detect_data_drift():
    try:
        # 1. Connect to MariaDB and load data
        encoded_password = quote_plus(DB_PASSWORD)
        engine = create_engine(f"mysql+pymysql://{DB_USER}:{encoded_password}@{DB_HOST}/{DB_NAME}")
        logger.info("Connected to MariaDB.")
        
        query = f"SELECT * FROM {TABLE_NAME}"
        df = pd.read_sql(query, engine)
        logger.info(f"Successfully loaded data from '{TABLE_NAME}'. Shape: {df.shape}")

    except Exception as e:
        logger.info(f"Error connecting to the database or loading data: {e}")
        raise

    # Add a source column for the chi2 test
    df['source'] = 'reference'
    df.loc[df.sample(frac=0.5, random_state=42).index, 'source'] = 'current'

    reference_df = df[df['source'] == 'reference']
    current_df = df[df['source'] == 'current']

    # Create directory for the report
    if not os.path.exists(DRIFT_REPORT_DIR):
        os.makedirs(DRIFT_REPORT_DIR)

    drift_report = ""

    numerical_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in numerical_cols:
        p_value = ks_2samp(reference_df[col], current_df[col]).pvalue
        drift_report += f"Feature: {col}, P-Value: {p_value:.5f}, Drift Detected: {p_value < 0.05}\n"

    for col in categorical_cols:
        contingency_table = pd.crosstab(df[col], df['source'])
        p_value = chi2_contingency(contingency_table)[1]
        drift_report += f"Feature: {col}, P-Value: {p_value:.5f}, Drift Detected: {p_value < 0.05}\n"

    report_path = os.path.join(DRIFT_REPORT_DIR, "data_drift_report.txt")
    with open(report_path, "w") as f:
        f.write(drift_report)

    # Log the report to MLflow
    with mlflow.start_run() as run:
        mlflow.log_artifact(report_path, artifact_path="drift_reports")
        logger.info(f"Data drift report logged to MLflow run: {run.info.run_id}")

if __name__ == "__main__":
    detect_data_drift()
