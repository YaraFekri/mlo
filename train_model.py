
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data import process_data, load_data
from src.model import train_model, save_model, save_encoder, save_lb

# Load data
data = load_data("starter/data/census.csv")

# Clean column names - remove leading/trailing spaces
data.columns = data.columns.str.strip()

# Define categorical features and target
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

# Process data
X, y, encoder, lb = process_data(
    data,
    categorical_features=categorical_features,
    label="salary",
    training=True
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Train model
model = train_model(X_train, y_train)

# Save model and encoder/lb
save_model(model, "model/model.pkl")
save_encoder(encoder, "model/encoder.pkl")
save_lb(lb, "model/lb.pkl")

print("Model, encoder, and label binarizer saved successfully.")


