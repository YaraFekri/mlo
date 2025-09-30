
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, precision_score, recall_score

def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Args:
        X_train (np.array): Training data features.
        y_train (np.array): Training data labels.

    Returns:
        sklearn.ensemble.RandomForestClassifier: The trained model.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def compute_model_metrics(y, preds):
    """
    Computes the performance metrics for the model.

    Args:
        y (np.array): True labels.
        preds (np.array): Predicted labels.

    Returns:
        tuple: A tuple containing precision, recall, and fbeta score.
    """
    fbeta = fbeta_score(y, preds, beta=0.5, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta

def inference(model, X):
    """
    Runs model inferences and returns the predictions.

    Args:
        model (sklearn.ensemble.RandomForestClassifier): Trained machine learning model.
        X (np.array): Data used for inference.

    Returns:
        np.array: Predictions from the model.
    """
    return model.predict(X)

def save_model(model, path):
    """
    Saves the trained model to a file.
    """
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(path):
    """
    Loads a trained model from a file.
    """
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def save_encoder(encoder, path):
    """
    Saves the fitted encoder to a file.
    """
    with open(path, 'wb') as f:
        pickle.dump(encoder, f)

def load_encoder(path):
    """
    Loads a fitted encoder from a file.
    """
    with open(path, 'rb') as f:
        encoder = pickle.load(f)
    return encoder

def save_lb(lb, path):
    """
    Saves the fitted LabelBinarizer to a file.
    """
    with open(path, 'wb') as f:
        pickle.dump(lb, f)

def load_lb(path):
    """
    Loads a fitted LabelBinarizer from a file.
    """
    with open(path, 'rb') as f:
        lb = pickle.load(f)
    return lb


