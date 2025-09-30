
import pandas as pd
from src.data import process_data, load_data
from src.model import load_model, load_encoder, load_lb, inference, compute_model_metrics

def compute_slice_performance(data, categorical_features, model, encoder, lb, output_file="slice_output.txt"):
    """
    Computes and prints model performance metrics on slices of data based on categorical features.
    """
    with open(output_file, "w") as f:
        f.write("Model Performance on Data Slices:\n\n")
        for feature in categorical_features:
            f.write(f"Feature: {feature}\n")
            # Get unique values for the current feature after stripping spaces
            unique_values = data[feature].str.strip().unique()
            for value in unique_values:
                slice_data = data[data[feature].str.strip() == value]
                
                # Process slice data for inference
                X_slice, y_slice, _, _ = process_data(
                    slice_data,
                    categorical_features=categorical_features,
                    label="salary",
                    training=False,
                    encoder=encoder,
                    lb=lb
                )
                
                if len(X_slice) == 0:
                    f.write(f"  Value: {value} - No data in slice\n")
                    continue

                preds = inference(model, X_slice)
                precision, recall, fbeta = compute_model_metrics(y_slice, preds)
                
                f.write(f"  Value: {value}\n")
                f.write(f"    Precision: {precision:.4f}\n")
                f.write(f"    Recall: {recall:.4f}\n")
                f.write(f"    F-beta: {fbeta:.4f}\n\n")

if __name__ == "__main__":
    # Load data
    data = load_data("starter/data/census.csv")
    # Clean column names - remove leading/trailing spaces
    data.columns = data.columns.str.strip()

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

    # Load trained model and encoders
    model = load_model("model/model.pkl")
    encoder = load_encoder("model/encoder.pkl")
    lb = load_lb("model/lb.pkl")

    # Compute and save slice performance
    compute_slice_performance(data, categorical_features, model, encoder, lb)
    print("Slice performance analysis complete. Output saved to slice_output.txt")


