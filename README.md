# Adult Census Income Prediction Project

This project implements a machine learning pipeline to predict income levels based on census data, including data processing, model training, API creation, and CI/CD setup.

## Project Structure

```
new_ml_project/
├── .github/
│   └── workflows/
│       └── ci.yml           # GitHub Actions CI workflow
├── api/
│   └── main.py              # FastAPI application
├── model/
│   ├── encoder.pkl          # Trained OneHotEncoder
│   ├── lb.pkl               # Trained LabelBinarizer
│   └── model.pkl            # Trained Random Forest model
├── notebooks/
│   └── census_eda.ipynb     # Exploratory Data Analysis notebook
├── src/
│   ├── __init__.py
│   ├── data.py              # Data processing functions
│   └── model.py             # Model training and inference functions
├── starter/
│   └── data/
│       └── census.csv       # Raw census data
├── tests/
│   ├── test_api.py          # Unit tests for API
│   └── test_model.py        # Unit tests for model functions
├── .gitignore               # Git ignore file
├── ModelCard.md             # Model documentation
├── Procfile                 # Process file for deployment
├── REPRODUCTION_STEPS.md    # Step-by-step guide to reproduce the project
├── README.md                # Project README
├── requirements.txt         # Python dependencies
├── render.yaml              # Render deployment configuration
├── runtime.txt              # Python runtime specification
├── slice_output.txt         # Model performance on data slices
├── slice_performance.py     # Script for slice performance analysis
├── train_model.py           # Script for model training
└── live_post.py             # Script to test live API
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd new_ml_project
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3.11 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Project

### 1. Train the Model

To train the machine learning model and save the artifacts (model, encoder, label binarizer):

```bash
python train_model.py
```

This will save `model.pkl`, `encoder.pkl`, and `lb.pkl` in the `model/` directory.

### 2. Analyze Slice Performance

To compute and save model performance metrics on different data slices:

```bash
python slice_performance.py
```

The results will be saved to `slice_output.txt`.

### 3. Run the API Locally

To start the FastAPI application locally:

```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000
```

Access the API at `http://localhost:8000` and the interactive documentation (Swagger UI) at `http://localhost:8000/docs`.

### 4. Run Tests

To run all unit tests for the model and API:

```bash
pytest tests/
```

To run code style checks:

```bash
flake8 .
```

### 5. Test Live API (after deployment)

After deploying the API, you can test it using the provided script:

```bash
python live_post.py
```

## Continuous Integration and Deployment (CI/CD)

This project is configured with GitHub Actions for Continuous Integration and Render for Continuous Deployment.

-   **CI (`.github/workflows/ci.yml`)**: Automatically runs `pytest` and `flake8` on every push and pull request to `main`/`master` branches.
-   **CD (`render.yaml`, `Procfile`, `runtime.txt`)**: Configuration files for deploying the FastAPI application to Render, enabling automatic deployments upon successful CI builds.

## Model Card

Refer to `ModelCard.md` for detailed documentation about the machine learning model, its intended use, performance, and ethical considerations.

## Reproduction Steps

For a detailed guide on how to reproduce this project from scratch, including environment setup, code implementation, and deployment, please refer to `REPRODUCTION_STEPS.md`.


