import pandas as pd
import mlflow
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import logging
import os
from urllib.parse import quote_plus
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MariaDB connection details
DB_HOST = "mariadb"
DB_USER = "manga_user"
DB_PASSWORD = "manga_password"
DB_NAME = "manga_db"
TABLE_NAME = "fact_manga"
PLOTS_DIR = "eda_manga_plots"

def run_eda():
    try:
        # 1. Connect to MariaDB and load data
        encoded_password = quote_plus(DB_PASSWORD)
        try:
            import pymysql
            logger.info("Successfully imported pymysql")
        except ImportError as e:
            logger.error(f"Failed to import pymysql: {e}")
            raise
        engine = create_engine(f"mysql+pymysql://{DB_USER}:{encoded_password}@{DB_HOST}/{DB_NAME}")
        logger.info("Connected to MariaDB.")
        
        query = """SELECT fm.*, dmi.type
FROM fact_manga fm
JOIN dim_manga_info dmi ON fm.manga_info_id = dmi.manga_info_id"""
        df = pd.read_sql(query, engine)
        logger.info(f"Successfully loaded data from '{TABLE_NAME}'. Shape: {df.shape}")

    except Exception as e:
        logger.info(f"Error connecting to the database or loading data: {e}")
        raise

    print(df.columns)
    # --- Print Tables to Console ---
    print("--- Data Preview (First 5 Rows) ---")
    print(df.head())
    print("\n" + "="*50 + "\n")
    print("--- Descriptive Statistics ---")
    print(df.describe())
    print("\n" + "="*50 + "\n")

    # Create directory for plots
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    # 2. Generate and save plots

    # Score distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['score'].dropna(), kde=True)
    plt.title('Score Distribution')
    plt.savefig(os.path.join(PLOTS_DIR, 'score_distribution.png'))
    plt.close()
    logger.info("Generated score distribution plot.")

    # Type vs. Score
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='type', y='score', data=df)
    plt.title('Type vs. Score')
    plt.savefig(os.path.join(PLOTS_DIR, 'type_vs_score.png'))
    plt.close()
    logger.info("Generated Type vs. Score plot.")

    # Top 10 Genres
    # plt.figure(figsize=(12, 6))
    # df['genres'].apply(lambda x: json.loads(x) if isinstance(x, str) else []).explode().value_counts().nlargest(10).plot(kind='bar')
    # plt.title('Top 10 Genres')
    # plt.ylabel('Count')
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig(os.path.join(PLOTS_DIR, 'top_10_genres.png'))
    # plt.close()
    # logger.info("Generated Top 10 Genres plot.")

    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    numerical_cols = df.select_dtypes(include=['number']).columns
    corr = df[numerical_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig(os.path.join(PLOTS_DIR, 'correlation_heatmap.png'))
    plt.close()
    logger.info("Generated correlation heatmap.")

    # 3. Log plots to MLflow
    with mlflow.start_run() as run:
        mlflow.log_artifacts(PLOTS_DIR, artifact_path="eda_manga_plots")
        logger.info(f"EDA plots logged to MLflow run: {run.info.run_id}")

if __name__ == "__main__":
    run_eda()
