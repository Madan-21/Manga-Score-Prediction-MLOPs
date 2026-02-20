from __future__ import annotations

import pendulum
import sys
import os

from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
# from airflow.operators.bash import BashOperator # NEW IMPORT # REMOVED
import docker # NEW IMPORT
from airflow.utils.dates import days_ago

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_ingestion import ingest_data
from src.data_preprocessing import preprocess_data
from src.feature_engineering import feature_engineering
from src.model_training import model_training
from src.model_evaluation import model_evaluation
from src.data_validation import validate_data # NEW IMPORT
from src.model_monitoring import run_model_monitoring # NEW IMPORT
from src.database_utils import get_db_engine, create_star_schema # <-- NEW IMPORT


# Define the Python callable for model_evaluation that pulls XCom
def _evaluate_model(ti):
    training_run_id = ti.xcom_pull(task_ids='model_training', key='return_value')
    if not training_run_id:
        raise Exception("MLflow training run_id not found in XComs.")
    model_evaluation(training_run_id=training_run_id)

# Define the Python callable for creating the star schema
def _create_star_schema():
    engine = get_db_engine()
    create_star_schema(engine)

# Define the Python callable for data validation
def _validate_processed_data():
    validate_data(file_path='/opt/airflow/data/processed/manga_processed.parquet')

# Define the Python callable for model monitoring
def _run_model_monitoring():
    run_model_monitoring()

# Define the Python callable for updating the latest run_id for model deployment
def _update_deployed_model_id(ti):
    import os
    training_run_id = ti.xcom_pull(task_ids='model_training', key='return_value')
    if not training_run_id:
        raise Exception("MLflow training run_id not found in XComs.")

    project_root = '/opt/airflow'
    run_id_path = os.path.join(project_root, 'data', 'processed', 'latest_run_id.txt')

    with open(run_id_path, 'w') as f:
        f.write(training_run_id)
    print(f"Updated latest_run_id.txt with run_id: {training_run_id}")

# Define the Python callable for restarting the FastAPI service
def _restart_fastapi_service():
    client = docker.from_env()
    try:
        container = client.containers.get('fastapi_app')
        container.restart()
        print("FastAPI app container restarted successfully.")
    except docker.errors.NotFound:
        print("FastAPI app container not found. It might not be running.")
    except Exception as e:
        print(f"Error restarting FastAPI app container: {e}")
        raise


with DAG(
    dag_id="manga_prediction_pipeline",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    schedule_interval=None,
    tags=["manga", "prediction"],
) as dag:
    start_pipeline = PythonOperator(
        task_id="start_pipeline",
        python_callable=lambda: print("Pipeline started!"),
    )

    create_schema_task = PythonOperator( # <-- NEW TASK
        task_id="create_star_schema",
        python_callable=_create_star_schema,
    )

    data_ingestion_task = PythonOperator(
        task_id="data_ingestion",
        python_callable=ingest_data,
    )

    data_preprocessing_task = PythonOperator(
        task_id="data_preprocessing",
        python_callable=preprocess_data,
    )

    data_validation_task = PythonOperator( # NEW TASK
        task_id="data_validation",
        python_callable=_validate_processed_data,
    )

    feature_engineering_task = PythonOperator(
        task_id="feature_engineering",
        python_callable=feature_engineering,
    )

    model_training_task = PythonOperator(
        task_id="model_training",
        python_callable=model_training,
    )

    model_evaluation_task = PythonOperator(
        task_id="model_evaluation",
        python_callable=_evaluate_model,
    )

    update_deployed_model_id = PythonOperator(
        task_id="update_deployed_model_id",
        python_callable=_update_deployed_model_id,
    )

    restart_model_serving_service = PythonOperator(
        task_id="restart_model_serving_service",
        python_callable=_restart_fastapi_service,
    )

    model_monitoring_task = PythonOperator(
        task_id="model_monitoring",
        python_callable=_run_model_monitoring,
    )

    end_pipeline = PythonOperator(
        task_id="end_pipeline",
        python_callable=lambda: print("Pipeline finished!"),
    )

    # Define the task dependencies
    start_pipeline >> create_schema_task >> data_ingestion_task >> data_preprocessing_task >> data_validation_task >> feature_engineering_task
    feature_engineering_task >> model_training_task >> model_evaluation_task >> update_deployed_model_id >> restart_model_serving_service >> model_monitoring_task >> end_pipeline
