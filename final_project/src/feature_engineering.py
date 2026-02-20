import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def feature_engineering():
    """
    Reads the preprocessed data, engineers features, and saves the final
    dataset for model training.
    """
    project_root = '/opt/airflow'
    processed_data_path = os.path.join(project_root, 'data', 'processed', 'manga_processed.parquet')
    features_dir = os.path.join(project_root, 'data', 'features')
    features_path = os.path.join(features_dir, 'manga_features.parquet')

    print(f"Reading processed data from {processed_data_path}")
    df = pd.read_parquet(processed_data_path)
    print("Processed data read successfully. Shape:", df.shape)

    # --- Feature Engineering Steps ---

    # 1. Identify column types
    numerical_cols = ['members', 'favorites', 'scored_by', 'volumes', 'chapters']
    categorical_cols = ['type']
    
    # Make sure all numerical columns exist and are numeric
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    df[numerical_cols] = df[numerical_cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=numerical_cols, inplace=True)

    # Fill any remaining NaNs with the median
    for col in numerical_cols:
        df[col].fillna(df[col].median(), inplace=True)


    print(f"Scaling numerical columns: {numerical_cols}")
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    print(f"One-hot encoding categorical columns: {categorical_cols}")
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # --- End of Feature Engineering ---

    print("Feature engineering complete.")
    
    # Ensure the features directory exists
    os.makedirs(features_dir, exist_ok=True)

    print(f"Saving features to {features_path}")
    df.to_parquet(features_path, index=False)
    print("Features saved successfully.")

if __name__ == '__main__':
    feature_engineering()
