import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import mysql.connector
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

mlflow.set_tracking_uri("file:///home/ashura/airflow/dags/mlruns")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MariaDB connection details
DB_HOST = "127.0.0.1"
DB_USER = "root"
DB_PASSWORD = "admin@123"
DB_NAME = "airflow_db"
TABLE_NAME = "manga_data"
ARTIFACTS_DIR = "training_artifacts_manga"

def train_model():
    try:
        # 1. Connect to MariaDB and load data
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        logger.info("Connected to MariaDB.")
        
        query = f"SELECT * FROM {TABLE_NAME}"
        df = pd.read_sql(query, conn)
        logger.info(f"Successfully loaded data from '{TABLE_NAME}'. Shape: {df.shape}")

    except mysql.connector.Error as err:
        logger.error(f"MariaDB error: {err}")
        raise
    finally:
        if 'conn' in locals() and conn.is_connected():
            conn.close()
            logger.info("MariaDB connection closed.")

    # Preprocess the data
    df = df.dropna(subset=['score', 'scored_by', 'members', 'favorites', 'type'])
    df = pd.get_dummies(df, columns=['type'], drop_first=True)

    # Select features and target
    features = ['scored_by', 'members', 'favorites'] + [col for col in df.columns if 'type_' in col]
    target = 'score'

    X = df[features]
    y = df[target]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create directory for artifacts
    if not os.path.exists(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR)

    # Save the scaler
    scaler_path = os.path.join(ARTIFACTS_DIR, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)

    # Start an MLflow run
    with mlflow.start_run() as run:
        mlflow.set_tag("run_type", "training_manga_score")
        # Create and train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = model.predict(X_test_scaled)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log parameters and metrics
        mlflow.log_param('features', features)
        mlflow.log_metric('mean_squared_error', mse)
        mlflow.log_metric('r2_score', r2)
        mlflow.log_param('scaler', 'StandardScaler')
        mlflow.log_param('n_estimators', 100)

        # --- Generate and Log Plots ---

        # Actual vs. Predicted
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_test, y=y_pred)
        plt.xlabel("Actual Score")
        plt.ylabel("Predicted Score")
        plt.title("Actual vs. Predicted Score")
        actual_vs_predicted_path = os.path.join(ARTIFACTS_DIR, 'actual_vs_predicted.png')
        plt.savefig(actual_vs_predicted_path)
        plt.close()
        mlflow.log_artifact(actual_vs_predicted_path)

        # Feature Importance
        feature_importances = pd.DataFrame({'feature': features, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importances)
        plt.title('Feature Importance')
        feature_importance_path = os.path.join(ARTIFACTS_DIR, 'feature_importance.png')
        plt.tight_layout()
        plt.savefig(feature_importance_path)
        plt.close()
        mlflow.log_artifact(feature_importance_path)

        # Log the scaler
        mlflow.log_artifact(scaler_path)

        # Log the model
        mlflow.sklearn.log_model(model, 'model')
        
        logger.info(f"Model training complete. MSE: {mse}, R2: {r2}")
        logger.info(f"Run ID: {run.info.run_id}")


if __name__ == "__main__":
    train_model()
