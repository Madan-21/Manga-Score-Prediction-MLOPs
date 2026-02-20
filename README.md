# Manga Score Prediction MLOps Project

This project implements an end-to-end MLOps pipeline for predicting manga scores using machine learning. It includes data ingestion, preprocessing, feature engineering, model training, evaluation, monitoring, and deployment components, all orchestrated for production-ready workflows.

## Features
- **Data Ingestion & Preprocessing:** Automated pipelines for collecting and cleaning manga data.
- **Feature Engineering:** Extraction and transformation of features for model input.
- **Model Training & Evaluation:** Training machine learning models and evaluating their performance.
- **Model Monitoring:** Automated monitoring for data drift and model performance.
- **Deployment:** Dockerized setup for reproducible environments and easy deployment.
- **Logging & Tracking:** Integrated logging and experiment tracking (e.g., MLflow).

## Project Structure
```
final_project/
├── dags/                  # Airflow DAGs for orchestration
├── data/                  # Raw, processed, and staged data
├── logs/                  # Pipeline and model logs
├── mlflow_data/           # MLflow artifacts and backend
├── plugins/               # Custom Airflow plugins
├── src/                   # Source code for all pipeline steps
│   ├── app.py
│   ├── data_ingestion.py
│   ├── ...
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker image for pipeline
├── Dockerfile.fastapi     # Docker image for FastAPI service
├── docker-compose.yml     # Multi-container orchestration
└── .gitignore             # Git ignore rules
```

## Getting Started
1. **Clone the repository:**
   ```sh
   git clone https://github.com/Madan-21/Manga-Score-Prediction-MLOPs.git
   cd Manga-Score-Prediction-MLOPs/final_project
   ```
2. **Build and run with Docker Compose:**
   ```sh
   docker-compose up --build
   ```
3. **Access services:**
   - Airflow UI: http://localhost:8080
   - FastAPI: http://localhost:8000
   - MLflow Tracking: http://localhost:5000

## Notes
- Large files and folders (data, logs, MLflow artifacts, reports) are excluded from version control via `.gitignore`.
- For files >50MB, consider using [Git LFS](https://git-lfs.github.com/).
- For reproducibility, use the provided Dockerfiles and `requirements.txt`.

## Dataset
The manga dataset used in this project was sourced from [Kaggle: Manga Scores Dataset](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database) or a similar public dataset. Please refer to the `data/manga.csv` file for details.

## Author
Done by Madan Pandey.
