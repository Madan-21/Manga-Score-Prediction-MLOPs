import pandas as pd
import os
import json
from sqlalchemy import text
from src.database_utils import get_db_engine

def ingest_data():
    """
    Reads the manga dataset from a CSV file and inserts it into MariaDB star schema.
    """
    project_root = '/opt/airflow'
    raw_data_path = os.path.join(project_root, 'data', 'manga.csv')

    print(f"Reading data from {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    print("Data read successfully. Shape:", df.shape)

    engine = get_db_engine()

    with engine.connect() as connection:
        print("Inserting into dim_manga_info...")
        
        # Dynamically select columns that exist in the DataFrame
        manga_info_cols = [
            'mal_id', 'title', 'title_english', 'title_japanese', 'title_synonyms',
            'synopsis', 'background', 'status', 'published_from', 'published_to',
            'url', 'type'
        ]
        
        # Add 'approved' and 'images' if they exist in the DataFrame
        if 'approved' in df.columns:
            manga_info_cols.append('approved')
        if 'images' in df.columns:
            manga_info_cols.append('images')

        manga_info_data = df[manga_info_cols].copy()
        manga_info_data['publishing'] = manga_info_data['status'].apply(lambda x: 1 if x == 'Publishing' else 0)
        
        # Handle 'approved' column if it was not in original df
        if 'approved' not in manga_info_data.columns:
            manga_info_data['approved'] = False # Default value
        else:
            manga_info_data['approved'] = manga_info_data['approved'].astype(bool)

        # Handle 'images' column if it was not in original df
        if 'images' not in manga_info_data.columns:
            manga_info_data['images'] = None 
  
        manga_info_data['published_from'] = pd.to_datetime(manga_info_data['published_from'], errors='coerce')
        manga_info_data['published_to'] = pd.to_datetime(manga_info_data['published_to'], errors='coerce')

        manga_info_data['published_from'] = manga_info_data['published_from'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else None)
        manga_info_data['published_to'] = manga_info_data['published_to'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else None)

        for index, row in manga_info_data.iterrows():
            try:
                params = row.to_dict()
                params['published_from'] = params['published_from'] if pd.notna(params['published_from']) else None
                params['published_to'] = params['published_to'] if pd.notna(params['published_to']) else None

                insert_stmt = text("""
                    INSERT INTO dim_manga_info (mal_id, title, title_english, title_japanese, title_synonyms,
                                                synopsis, background, status, publishing, published_from,
                                                published_to, approved, url, images, type)
                    VALUES (:mal_id, :title, :title_english, :title_japanese, :title_synonyms,
                            :synopsis, :background, :status, :publishing, :published_from,
                            :published_to, :approved, :url, :images, :type)
                    ON DUPLICATE KEY UPDATE
                        title=VALUES(title), title_english=VALUES(title_english),
                        title_japanese=VALUES(title_japanese), title_synonyms=VALUES(title_synonyms),
                        synopsis=VALUES(synopsis), background=VALUES(background),
                        status=VALUES(status), publishing=VALUES(publishing),
                        published_from=VALUES(published_from), published_to=VALUES(published_to),
                        approved=VALUES(approved), url=VALUES(url), images=VALUES(images), type=VALUES(type);
                """)
                connection.execute(insert_stmt, params)
            except Exception as e:
                print(f"Error inserting/updating manga_info for mal_id {row['mal_id']}: {e}")
        print("dim_manga_info populated.")

        # Get manga_info_id for later use
        manga_info_map = pd.read_sql("SELECT mal_id, manga_info_id FROM dim_manga_info", connection)
        df = df.merge(manga_info_map, on='mal_id', how='left')

        # Process and insert into dim_genres
        print("Inserting into dim_genres...")
        # Ensure 'genres' column is treated as string and then parsed
        def _safe_json_loads(s):
            try:
                return json.loads(s)
            except json.decoder.JSONDecodeError:
                return []
        def safe_json_loads(s):
            try:
                return json.loads(s) if isinstance(s, str) and s.startswith('[') else []
            except json.JSONDecodeError:
                return []

        unique_genres = df['genres'].astype(str).dropna().apply(safe_json_loads).explode().dropna().apply(lambda x: x['name'] if isinstance(x, dict) else x).unique()
        for genre_name in unique_genres:
            if genre_name:
                insert_stmt = text("INSERT IGNORE INTO dim_genres (genre_name) VALUES (:genre_name)")
                connection.execute(insert_stmt, {'genre_name': genre_name})
        print("dim_genres populated.")

        # Get genre_id for later use
        genre_map = pd.read_sql("SELECT genre_id, genre_name FROM dim_genres", connection)

        # Process and insert into dim_authors
        print("Inserting into dim_authors...")
        unique_authors = df['authors'].astype(str).dropna().apply(safe_json_loads).explode().dropna().apply(lambda x: x['name'] if isinstance(x, dict) else x).unique()
        for author_name in unique_authors:
            if author_name:
                insert_stmt = text("INSERT IGNORE INTO dim_authors (author_name) VALUES (:author_name)")
                connection.execute(insert_stmt, {'author_name': author_name})
        print("dim_authors populated.")

        # Get author_id for later use
        author_map = pd.read_sql("SELECT author_id, author_name FROM dim_authors", connection)

        # Process and insert into dim_demographics
        print("Inserting into dim_demographics...")
        unique_demographics = df['demographics'].astype(str).dropna().apply(safe_json_loads).explode().dropna().apply(lambda x: x['name'] if isinstance(x, dict) else x).unique()
        for demo_name in unique_demographics:
            if demo_name:
                insert_stmt = text("INSERT IGNORE INTO dim_demographics (demographic_name) VALUES (:demographic_name)")
                connection.execute(insert_stmt, {'demographic_name': demo_name})
        print("dim_demographics populated.")

        # Get demographic_id for later use
        demographic_map = pd.read_sql("SELECT demographic_id, demographic_name FROM dim_demographics", connection)

        # Process and insert into dim_serializations
        print("Inserting into dim_serializations...")
        unique_serializations = df['serializations'].astype(str).dropna().apply(safe_json_loads).explode().dropna().apply(lambda x: x['name'] if isinstance(x, dict) else x).unique()
        for ser_name in unique_serializations:
            if ser_name:
                insert_stmt = text("INSERT IGNORE INTO dim_serializations (serialization_name) VALUES (:serialization_name)")
                connection.execute(insert_stmt, {'serialization_name': ser_name})
        print("dim_serializations populated.")

        # Get serialization_id for later use
        serialization_map = pd.read_sql("SELECT serialization_id, serialization_name FROM dim_serializations", connection)

        # Insert into fact_manga
        print("Inserting into fact_manga...")
        for index, row in df.iterrows():
            primary_genre_id = None
            genres_list = safe_json_loads(row['genres'])
            if genres_list and isinstance(genres_list[0], dict) and 'name' in genres_list[0]:
                primary_genre_name = genres_list[0]['name']
                primary_genre_id = genre_map[genre_map['genre_name'] == primary_genre_name]['genre_id'].iloc[0]

            primary_author_id = None
            authors_list = safe_json_loads(row['authors'])
            if authors_list and isinstance(authors_list[0], dict) and 'name' in authors_list[0]:
                primary_author_name = authors_list[0]['name']
                primary_author_id = author_map[author_map['author_name'] == primary_author_name]['author_id'].iloc[0]

            primary_demographic_id = None
            demographics_list = safe_json_loads(row['demographics'])
            if demographics_list and isinstance(demographics_list[0], dict) and 'name' in demographics_list[0]:
                primary_demographic_name = demographics_list[0]['name']
                primary_demographic_id = demographic_map[demographic_map['demographic_name'] == primary_demographic_name]['demographic_id'].iloc[0]

            primary_serialization_id = None
            serializations_list = safe_json_loads(row['serializations'])
            if serializations_list and isinstance(serializations_list[0], dict) and 'name' in serializations_list[0]:
                primary_serialization_name = serializations_list[0]['name']
                primary_serialization_id = serialization_map[serialization_map['serialization_name'] == primary_serialization_name]['serialization_id'].iloc[0]

            insert_stmt = text("""
                INSERT INTO fact_manga (manga_info_id, score, scored_by, rank_val, popularity,
                                        members, favorites, volumes, chapters,
                                        primary_genre_id, primary_author_id,
                                        primary_demographic_id, primary_serialization_id)
                VALUES (:manga_info_id, :score, :scored_by, :rank_val, :popularity,
                        :members, :favorites, :volumes, :chapters,
                        :primary_genre_id, :primary_author_id,
                        :primary_demographic_id, :primary_serialization_id);
            """)
            connection.execute(insert_stmt, {
                'manga_info_id': row['manga_info_id'],
                'score': row['score'],
                'scored_by': row['scored_by'],
                'rank_val': row.get('rank'), # Use .get() to avoid KeyError
                'popularity': row.get('popularity'), # Use .get() to avoid KeyError
                'members': row['members'],
                'favorites': row['favorites'],
                'volumes': row['volumes'],
                'chapters': row['chapters'],
                'primary_genre_id': primary_genre_id,
                'primary_author_id': primary_author_id,
                'primary_demographic_id': primary_demographic_id,
                'primary_serialization_id': primary_serialization_id
            })
        print("fact_manga populated.")

        # Insert into manga_secondary_genres (bridge table)
        print("Inserting into manga_secondary_genres...")
        for index, row in df.iterrows():
            genres_list = safe_json_loads(row['genres'])
            if genres_list and len(genres_list) > 1:
                for genre_dict in genres_list[1:]: # Secondary genres
                    if isinstance(genre_dict, dict) and 'name' in genre_dict:
                        genre_name = genre_dict['name']
                        genre_id = genre_map[genre_map['genre_name'] == genre_name]['genre_id'].iloc[0]
                        insert_stmt = text("""
                            INSERT IGNORE INTO manga_secondary_genres (manga_info_id, genre_id)
                            VALUES (:manga_info_id, :genre_id);
                        """)
                        connection.execute(insert_stmt, {'manga_info_id': row['manga_info_id'], 'genre_id': genre_id})
        print("manga_secondary_genres populated.")

    print("Data ingestion to MariaDB complete.")

if __name__ == '__main__':
    ingest_data()
