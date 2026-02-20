import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def model_training():
    """
    Reads the feature-engineered data, trains a Random Forest Regressor model,
    logs metrics and the model to MLflow.
    Returns the MLflow run_id.
    """
    project_root = '/opt/airflow'
    features_path = os.path.join(project_root, 'data', 'processed', 'manga_processed.parquet')

    print(f"Reading feature-engineered data from {features_path}")
    df = pd.read_parquet(features_path)
    print("Feature-engineered data read successfully. Shape:", df.shape)

    # Define target and features
    target_column = 'score'
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    features = numeric_cols
    
    X = df[features]
    y = df[target_column]

    # Fill NaN values with the median of each column
    for col in X.columns:
        X[col].fillna(X[col].median(), inplace=True)

    # Split data
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # Save training columns
    training_columns_path = os.path.join(project_root, 'data', 'processed', 'training_columns.txt')
    with open(training_columns_path, 'w') as f:
        f.write(str(X_train.columns.tolist()))

    mlflow.set_experiment("manga_prediction") # <-- MOVED HERE

    # MLflow tracking
    with mlflow.start_run(run_name="RandomForest_Manga_Prediction") as run:
        # Model training
        print("Training Random Forest Regressor model...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        print("Model training complete.")

        # Predictions
        y_pred = model.predict(X_test)

        # Evaluate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False) # RMSE
        r2 = r2_score(y_test, y_pred)

        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2 Score: {r2:.4f}")

        # Log parameters and metrics to MLflow
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)

        # Log the model
        mlflow.sklearn.log_model(model, "random_forest_model")
        print("Model and metrics logged to MLflow.")

        # Save the run_id for later use
        run_id_path = os.path.join(project_root, 'data', 'processed', 'latest_run_id.txt')
        with open(run_id_path, 'w') as f:
            f.write(run.info.run_id)
        
        return run.info.run_id

if __name__ == '__main__':
    model_training()