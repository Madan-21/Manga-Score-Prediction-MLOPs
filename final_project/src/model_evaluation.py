import pandas as pd
import os
import mlflow
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import mlflow.pyfunc

def model_evaluation(training_run_id: str):
    """
    Loads the trained model from MLflow, evaluates it on the test set,
    and logs evaluation metrics to MLflow.
    """
    project_root = '/opt/airflow'
    features_path = os.path.join(project_root, 'data', 'features', 'manga_features.parquet')

    print(f"Reading feature-engineered data from {features_path} for evaluation split")
    df = pd.read_parquet(features_path)
    
    target_column = 'score'
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    features = numeric_cols
    
    X = df[features]
    y = df[target_column]

    # Split data again to get the same test set as in training
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Load training columns
    training_columns_path = os.path.join(project_root, 'data', 'processed', 'training_columns.txt')
    with open(training_columns_path, 'r') as f:
        training_columns = eval(f.read())

    # Align columns
    X_test = X_test.reindex(columns=training_columns, fill_value=0)

    mlflow.set_experiment("manga_prediction") # <-- MOVED HERE

    # MLflow tracking
    with mlflow.start_run(run_name="Model_Evaluation"):
        # Load the trained model using the provided run_id
        model_uri = f"runs:/{training_run_id}/random_forest_model"
        
        print(f"Loading model from MLflow URI: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        print("Model loaded successfully.")

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False) # RMSE
        r2 = r2_score(y_test, y_pred)

        print(f"Evaluation Metrics:")
        print(f"  MAE: {mae:.4f}")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R2 Score: {r2:.4f}")

        # Log metrics to MLflow
        mlflow.log_metric("eval_mae", mae)
        mlflow.log_metric("eval_mse", mse)
        mlflow.log_metric("eval_rmse", rmse)
        mlflow.log_metric("eval_r2_score", r2)
        print("Evaluation metrics logged to MLflow.")

if __name__ == '__main__':
    # This part is for local testing, not used by Airflow
    # You would need to provide a valid run_id here for local testing
    print("This script is designed to be run as part of an Airflow DAG.")
    print("For local testing, you would need to provide a training_run_id.")
    # Example: model_evaluation(training_run_id="your_mlflow_run_id_here")
