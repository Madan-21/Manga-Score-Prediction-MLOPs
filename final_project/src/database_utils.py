import sqlalchemy
from sqlalchemy import create_engine, text
import pandas as pd
import os

# Database connection details (from docker-compose.yml)
DB_USER = os.getenv("MARIADB_USER", "manga_user")
DB_PASSWORD = os.getenv("MARIADB_PASSWORD", "manga_password")
DB_HOST = os.getenv("MARIADB_HOST", "mariadb") # Service name in docker-compose
DB_PORT = os.getenv("MARIADB_PORT", "3306")
DB_NAME = os.getenv("MARIADB_DATABASE", "manga_db")

DATABASE_URL = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def get_db_engine():
    """Establishes and returns a SQLAlchemy engine for MariaDB."""
    print(f"Attempting to connect to MariaDB at {DB_HOST}:{DB_PORT}/{DB_NAME}")
    engine = create_engine(DATABASE_URL)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        print("Successfully connected to MariaDB.")
        return engine
    except Exception as e:
        print(f"Error connecting to MariaDB: {e}")
        raise

def create_star_schema(engine):
    """Creates the star schema tables in MariaDB."""
    print("Creating star schema tables if they don't exist...")
    with engine.connect() as connection:
        # Drop tables in reverse order of dependency to avoid foreign key issues
        connection.execute(text("DROP TABLE IF EXISTS manga_secondary_genres"))
        connection.execute(text("DROP TABLE IF EXISTS fact_manga"))
        connection.execute(text("DROP TABLE IF EXISTS dim_genres"))
        connection.execute(text("DROP TABLE IF EXISTS dim_authors"))
        connection.execute(text("DROP TABLE IF EXISTS dim_demographics"))
        connection.execute(text("DROP TABLE IF EXISTS dim_serializations"))
        connection.execute(text("DROP TABLE IF EXISTS dim_manga_info")) # Added this drop
        # connection.commit() # REMOVED THIS LINE

        # Dimension Tables
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS dim_genres (
                genre_id INT AUTO_INCREMENT PRIMARY KEY,
                genre_name VARCHAR(255) UNIQUE
            );
        """))
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS dim_authors (
                author_id INT AUTO_INCREMENT PRIMARY KEY,
                author_name VARCHAR(255) UNIQUE
            );
        """))
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS dim_demographics (
                demographic_id INT AUTO_INCREMENT PRIMARY KEY,
                demographic_name VARCHAR(255) UNIQUE
            );
        """))
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS dim_serializations (
                serialization_id INT AUTO_INCREMENT PRIMARY KEY,
                serialization_name VARCHAR(255) UNIQUE
            );
        """))
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS dim_manga_info (
                manga_info_id INT AUTO_INCREMENT PRIMARY KEY,
                mal_id INT UNIQUE,
                title VARCHAR(512),
                title_english VARCHAR(512),
                title_japanese VARCHAR(512),
                title_synonyms TEXT,
                synopsis TEXT,
                background TEXT,
                status VARCHAR(255),
                type VARCHAR(255),
                publishing BOOLEAN,
                published_from DATE,
                published_to DATE,
                approved BOOLEAN,
                url VARCHAR(1024),
                images TEXT
            );
        """))

        # Fact Table
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS fact_manga (
                fact_id INT AUTO_INCREMENT PRIMARY KEY,
                manga_info_id INT,
                score DECIMAL(3,2),
                scored_by INT,
                rank_val INT,
                popularity INT,
                members INT,
                favorites INT,
                volumes INT,
                chapters INT,
                primary_genre_id INT,
                primary_author_id INT,
                primary_demographic_id INT,
                primary_serialization_id INT,
                FOREIGN KEY (manga_info_id) REFERENCES dim_manga_info(manga_info_id),
                FOREIGN KEY (primary_genre_id) REFERENCES dim_genres(genre_id),
                FOREIGN KEY (primary_author_id) REFERENCES dim_authors(author_id),
                FOREIGN KEY (primary_demographic_id) REFERENCES dim_demographics(demographic_id),
                FOREIGN KEY (primary_serialization_id) REFERENCES dim_serializations(serialization_id)
            );
        """))

        # Bridge Table for many-to-many genres
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS manga_secondary_genres (
                manga_info_id INT,
                genre_id INT,
                PRIMARY KEY (manga_info_id, genre_id),
                FOREIGN KEY (manga_info_id) REFERENCES dim_manga_info(manga_info_id),
                FOREIGN KEY (genre_id) REFERENCES dim_genres(genre_id)
            );
        """))
        # connection.commit() # REMOVED THIS LINE
    print("Star schema tables created successfully.")

if __name__ == '__main__':
    try:
        engine = get_db_engine()
        create_star_schema(engine)
    except Exception as e:
        print(f"Failed to initialize database: {e}")