import os
import logging
import pandas as pd
# import mysql.connector
from typing import Dict, List
from tabulate import tabulate
import argparse

# Setup logging for Airflow
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MariaDB connection details
# DB_HOST = "mariadb"
# DB_USER = "manga_user"
# DB_PASSWORD = "manga_password"
# DB_NAME = "manga_db"

def validate_data(file_path: str) -> Dict[str, List[str]]:
    """
    Validates data in a Parquet file for an Airflow task.
    Returns a dictionary with validation results.
    Raises ValueError if validation fails.
    """
    validation_results = {"errors": [], "success": True}
    validation_summary = []  # For table output

    try:
        # Load data from Parquet file
        df = pd.read_parquet(file_path)
        logger.info(f"Data read successfully from {file_path}. Shape: {df.shape}")
        logger.info("Data preview:")
        logger.info(df.head().to_string())

        # Define columns that are allowed to have missing values (based on previous logs)
        allowed_missing_cols = [
            'title_english', 'title_japanese', 'title_synonyms', 'synopsis', 'background',
            'published_from', 'published_to', 'images', 'rank_val', 'popularity',
            'primary_genre', 'primary_author', 'primary_demographic', 'primary_serialization',
            'volumes', 'chapters' # Allow missing for these as well for now
        ]

        # Validation 2: Check for missing values in *critical* columns
        critical_cols_with_missing = []
        for col in df.columns:
            if col not in allowed_missing_cols and df[col].isnull().any():
                critical_cols_with_missing.append(col)

        if critical_cols_with_missing:
            error_msg = f"Missing values found in critical columns: {critical_cols_with_missing}"
            logger.error(error_msg)
            validation_results["errors"].append(error_msg)
            validation_results["success"] = False
            validation_summary.append(["Missing Values (Critical)", "Failed", error_msg])
        else:
            validation_summary.append(["Missing Values (Critical)", "Passed", "No missing values in critical columns"])

        # Validation 3: Check if all required columns exist
        required_columns = ['manga_info_id', 'mal_id', 'title', 'score', 'scored_by', 'members', 'favorites']
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            error_msg = f"Missing required columns: {missing_cols}"
            logger.error(error_msg)
            validation_results["errors"].append(error_msg)
            validation_results["success"] = False
            validation_summary.append(["Required Columns", "Failed", error_msg])
        else:
            validation_summary.append(["Required Columns", "Passed", "All required columns present"])

        # Validation 4: Check manga_info_id uniqueness
        if "manga_info_id" in df.columns and df["manga_info_id"].duplicated().any():
            error_msg = f"Found {df['manga_info_id'].duplicated().sum()} duplicate manga_info_ids"
            logger.error(error_msg)
            validation_results["errors"].append(error_msg)
            validation_results["success"] = False
            validation_summary.append(["manga_info_id Uniqueness", "Failed", error_msg])
        else:
            validation_summary.append(["manga_info_id Uniqueness", "Passed", "All manga_info_ids are unique"])

        # Validation 5: Check score is within a valid range (e.g., 0-10)
        if "score" in df.columns:
            df['score'] = pd.to_numeric(df['score'], errors='coerce')
            if df['score'].isnull().any() or (df["score"] < 0).any() or (df["score"] > 10).any():
                error_msg = "Invalid score values: must be between 0 and 10"
                logger.error(error_msg)
                validation_results["errors"].append(error_msg)
                validation_results["success"] = False
                validation_summary.append(["Score Range", "Failed", error_msg])
            else:
                validation_summary.append(["Score Range", "Passed", "All score values between 0 and 10"])

        # Validation 6: Check scored_by, members, favorites are non-negative integers
        for col in ["scored_by", "members", "favorites"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isnull().any() or (df[col] < 0).any() or (df[col] % 1 != 0).any():
                    error_msg = f"Invalid {col} values: must be non-negative integers"
                    logger.error(error_msg)
                    validation_results["errors"].append(error_msg)
                    validation_results["success"] = False
                    validation_summary.append([f"{col} Values", "Failed", error_msg])
                else:
                    validation_summary.append([f"{col} Values", "Passed", f"All {col} values are non-negative integers"])

        # Validation 7: Check for duplicate rows
        if df.duplicated().any():
            error_msg = f"Found {df.duplicated().sum()} duplicate rows"
            logger.error(error_msg)
            validation_results["errors"].append(error_msg)
            validation_results["success"] = False
            validation_summary.append(["Duplicates", "Failed", error_msg])
        else:
            validation_summary.append(["Duplicates", "Passed", "No duplicate rows"])

        # Log validation summary as a table
        logger.info(f"\nValidation Summary for file {file_path}:")
        logger.info(tabulate(validation_summary, headers=["Check", "Status", "Details"], tablefmt="grid"))

        # Stop if validation fails
        if not validation_results["success"]:
            raise ValueError(f"Data validation failed for file {file_path}: " + "; ".join(validation_results["errors"]))

    except ValueError as e:
        logger.error(f"Validation failed: {str(e)}")
        raise
    except Exception as e:
        error_msg = f"Error processing data: {str(e)}"
        logger.error(error_msg)
        validation_results["errors"].append(error_msg)
        validation_results["success"] = False
        raise
    finally:
        pass # No connection to close

    logger.info(f"Data validation for file {file_path} successful.")
    return validation_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate data in a Parquet file.')
    parser.add_argument('--file_path', required=True, help='Path to the Parquet file to validate.')
    args = parser.parse_args()

    try:
        result = validate_data(args.file_path)
        logger.info(f"Validation result for file {args.file_path}: {result}")
    except (ValueError) as e:
        logger.error(f"Operation failed: {str(e)}")
        raise
