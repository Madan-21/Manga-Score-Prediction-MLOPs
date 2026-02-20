import os
import pandas as pd
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel, Field # Import Field

# Define the input data model
class MangaFeatures(BaseModel):
    manga_info_id: int = Field(..., description="Unique identifier for manga information.")
    mal_id: int = Field(..., description="MyAnimeList ID of the manga.")
    publishing: bool = Field(..., description="True if the manga is currently publishing, False otherwise.")
    approved: bool = Field(..., description="True if the manga is approved, False otherwise.")
    scored_by: int = Field(..., description="Number of users who have scored the manga.")
    members: int = Field(..., description="Number of members who have added the manga to their list.")
    favorites: int = Field(..., description="Number of users who have favorited the manga.")
    volumes: int = Field(..., description="Number of volumes in the manga.")
    chapters: int = Field(..., description="Number of chapters in the manga.")

    class Config:
        schema_extra = {
            "example": {
                "manga_info_id": 1,
                "mal_id": 2,
                "publishing": True,
                "approved": True,
                "scored_by": 267095,
                "members": 548371,
                "favorites": 103266,
                "volumes": 41,
                "chapters": 364
            }
        }

# Initialize the FastAPI app
app = FastAPI()

# Load the model
project_root = '/opt/airflow'
run_id_path = os.path.join(project_root, 'data', 'processed', 'latest_run_id.txt')
with open(run_id_path, 'r') as f:
    run_id = f.read()

model_uri = f"runs:/{run_id}/random_forest_model"
model = mlflow.pyfunc.load_model(model_uri)

@app.post("/predict", response_model=dict, summary="Predict manga score based on features")
def predict(features: MangaFeatures):
    """
    Receives manga features and returns a score prediction.
    """
    # Convert input to DataFrame
    feature_df = pd.DataFrame([features.dict()])

    # Predict
    prediction = model.predict(feature_df)

    return {"predicted_score": prediction[0]}

@app.get("/")
def read_root():
    return {"message": "Manga Score Prediction API"}
