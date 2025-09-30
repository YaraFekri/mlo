from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from src.data import process_data
from src.model import load_model, load_encoder, load_lb, inference

# Initialize FastAPI app
app = FastAPI(title="Adult Census Income Prediction API", version="1.0.0")

# Load the trained model and encoders
model = load_model("model/model.pkl")
encoder = load_encoder("model/encoder.pkl")
lb = load_lb("model/lb.pkl")

# Define categorical features
categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Pydantic model for input data
class CensusData(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., alias="education-num", example=13)
    marital_status: str = Field(..., alias="marital-status", example="Never-married")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", example=2174)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=40)
    native_country: str = Field(..., alias="native-country", example="United-States")

    class Config:
        allow_population_by_field_name = True

@app.get("/")
def read_root():
    """
    Welcome message for the API.
    """
    return {"message": "Welcome to the Adult Census Income Prediction API!"}

@app.post("/predict")
def predict_income(data: CensusData):
    """
    Predict whether income exceeds $50K based on census data.
    """
    # Convert input data to DataFrame
    input_dict = data.dict(by_alias=True)
    input_df = pd.DataFrame([input_dict])
    
    # Process the data for inference
    X, _, _, _ = process_data(
        input_df,
        categorical_features=categorical_features,
        training=False,
        encoder=encoder,
        lb=lb
    )
    
    # Make prediction
    prediction = inference(model, X)
    
    # Convert prediction to human-readable format
    prediction_label = lb.inverse_transform(prediction)[0]
    
    return {
        "prediction": prediction_label,
        "prediction_binary": int(prediction[0])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

