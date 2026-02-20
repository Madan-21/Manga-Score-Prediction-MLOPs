import pandas as pd
import os
from src.database_utils import get_db_engine

def preprocess_data():
    """
    Reads data from MariaDB, preprocesses it, and saves it to a processed area.
    """
    processed_data_dir = '/opt/airflow/data/processed'
    processed_data_path = os.path.join(processed_data_dir, 'manga_processed.parquet')

    print("Connecting to MariaDB to read data...")
    engine = get_db_engine()
    
    query = """
        SELECT 
            mi.*,
            fm.score, fm.scored_by, fm.rank_val, fm.popularity, fm.members, fm.favorites,
            fm.volumes, fm.chapters,
            g.genre_name as primary_genre,
            a.author_name as primary_author,
            d.demographic_name as primary_demographic,
            s.serialization_name as primary_serialization
        FROM fact_manga fm
        JOIN dim_manga_info mi ON fm.manga_info_id = mi.manga_info_id
        LEFT JOIN dim_genres g ON fm.primary_genre_id = g.genre_id
        LEFT JOIN dim_authors a ON fm.primary_author_id = a.author_id
        LEFT JOIN dim_demographics d ON fm.primary_demographic_id = d.demographic_id
        LEFT JOIN dim_serializations s ON fm.primary_serialization_id = s.serialization_id;
    """
    
    df = pd.read_sql(query, engine)
    print("Data read successfully from MariaDB. Shape:", df.shape)

    # Drop duplicates based on mal_id to ensure unique manga entries
    df.drop_duplicates(subset=['mal_id'], inplace=True)
    print("Rows after dropping mal_id duplicates:", df.shape)

    # --- Preprocessing Steps ---

    # 1. Handle missing scores
    print(f"Converting 'score' to numeric and dropping NaNs. Original rows: {len(df)}")
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df.dropna(subset=['score'], inplace=True)
    print(f"Rows after dropping missing scores: {len(df)}")

    # --- End of Preprocessing ---

    print("Data preprocessing complete.")
    
    # Ensure the processed directory exists
    os.makedirs(processed_data_dir, exist_ok=True)

    print(f"Saving processed data to {processed_data_path}")
    df.to_parquet(processed_data_path, index=False)
    print("Processed data saved successfully.")

if __name__ == '__main__':
    preprocess_data()
