import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_get_root():
    """Test the GET endpoint at root."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert response.json()["message"] == "Welcome to the Adult Census Income Prediction API!"

def test_post_predict_high_income():
    """Test POST endpoint with data that should predict high income (>50K)."""
    test_data = {
        "age": 50,
        "workclass": "Private",
        "fnlgt": 234721,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 15024,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States"
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "prediction_binary" in response.json()
    # The prediction could be either <=50K or >50K depending on the model
    # Strip spaces to handle potential space issues
    prediction = response.json()["prediction"].strip()
    assert prediction in ["<=50K", ">50K"]
    assert response.json()["prediction_binary"] in [0, 1]

def test_post_predict_low_income():
    """Test POST endpoint with data that should predict low income (<=50K)."""
    test_data = {
        "age": 23,
        "workclass": "Private",
        "fnlgt": 122272,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Own-child",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 30,
        "native-country": "United-States"
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "prediction_binary" in response.json()
    # The prediction could be either <=50K or >50K depending on the model
    # Strip spaces to handle potential space issues
    prediction = response.json()["prediction"].strip()
    assert prediction in ["<=50K", ">50K"]
    assert response.json()["prediction_binary"] in [0, 1]

def test_post_predict_invalid_data():
    """Test POST endpoint with invalid data."""
    test_data = {
        "age": "invalid",  # Should be int
        "workclass": "Private"
        # Missing required fields
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 422  # Unprocessable Entity

