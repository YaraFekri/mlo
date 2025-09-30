# Machine Learning Project - Reproduction Steps

This document provides step-by-step instructions to reproduce the complete machine learning project from scratch.

## Prerequisites

- Python 3.11.0 or higher
- Git
- GitHub account
- Cloud deployment platform account (Render, Railway, or Heroku)

## Step 1: Environment Setup

### 1.1 Create Project Directory
```bash
mkdir new_ml_project
cd new_ml_project
```

### 1.2 Initialize Git Repository
```bash
git init
git config user.email "your-email@example.com"
git config user.name "Your Name"
```

### 1.3 Create Virtual Environment
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 1.4 Install Dependencies
Create `requirements.txt`:
```
pandas
numpy
scikit-learn
fastapi
uvicorn[standard]
pytest
flake8
matplotlib
seaborn
scipy
python-multipart
python-jose
passlib
httpx
requests
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Step 2: Data and Model Development

### 2.1 Obtain Dataset
Place `census.csv` in `starter/data/`.

### 2.2 Create Source Code Structure
```bash
mkdir src
mkdir tests
mkdir api
mkdir model
mkdir notebooks
```

### 2.3 Create Data Processing Module (`src/data.py`)
Implement functions for:
- `process_data()`: Clean and encode data
- `load_data()`: Load CSV data

### 2.4 Create Model Module (`src/model.py`)
Implement functions for:
- `train_model()`: Train Random Forest classifier
- `inference()`: Make predictions
- `compute_model_metrics()`: Calculate precision, recall, f-beta
- `save_model()`, `load_model()`: Model persistence
- `save_encoder()`, `load_encoder()`: Encoder persistence
- `save_lb()`, `load_lb()`: Label binarizer persistence

### 2.5 Create Training Script (`train_model.py`)
Script that:
- Loads and cleans data
- Processes features and target
- Trains model
- Saves model artifacts to `model/`

### 2.6 Create Slice Performance Analysis (`slice_performance.py`)
Script that:
- Loads trained model and data
- Computes metrics for each categorical feature value
- Outputs results to `slice_output.txt`

### 2.7 Create Unit Tests (`tests/test_model.py`)
Implement at least 6 tests covering:
- Data processing output types
- Model training functionality
- Inference functionality
- Metrics computation
- Data cleaning
- Missing value handling

## Step 3: API Development

### 3.1 Create FastAPI Application (`api/main.py`)
Implement:
- GET endpoint at root (`/`) returning welcome message
- POST endpoint (`/predict`) for model inference
- Pydantic model with field aliases for hyphenated columns
- Example data in Pydantic model

### 3.2 Create API Tests (`tests/test_api.py`)
Implement at least 4 tests:
- GET endpoint test (status code and content)
- POST endpoint test for high income prediction
- POST endpoint test for low income prediction
- POST endpoint test for invalid data

## Step 4: Continuous Integration

### 4.1 Create GitHub Actions Workflow
Create `.github/workflows/ci.yml`:
```yaml
name: CI
on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python 3.11
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install httpx
    - name: Run flake8
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    - name: Run pytest
      run: pytest
```

### 4.2 Create .gitignore
```
venv/
__pycache__/
*.pyc
*.pkl
slice_output.txt
```

## Step 5: Model Documentation

### 5.1 Create Model Card (`ModelCard.md`)
Document:
- Model details and version
- Intended use and users
- Training data and preprocessing
- Performance metrics
- Ethical considerations
- Limitations and recommendations

## Step 6: Deployment Configuration

### 6.1 Create Deployment Files
- `render.yaml`: Render platform configuration
- `Procfile`: Process specification
- `runtime.txt`: Python version specification

### 6.2 Create Live API Test Script (`live_post.py`)
Script that:
- Makes POST request to deployed API
- Prints status code and response
- Uses realistic test data

## Step 7: Testing and Validation

### 7.1 Run All Tests Locally
```bash
# Run model tests
pytest tests/test_model.py

# Run API tests
pytest tests/test_api.py

# Run all tests
pytest

# Check code style
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

### 7.2 Train Model and Generate Outputs
```bash
# Train the model
python train_model.py

# Generate slice performance analysis
python slice_performance.py

# Test API locally
python api/main.py &
# Test in another terminal
python live_post.py
```

## Step 8: Git Operations

### 8.1 Commit All Files
```bash
git add .
git commit -m "Initial commit with ML model, API, and CI setup"
```

### 8.2 Push to GitHub
```bash
# Create repository on GitHub first, then:
git remote add origin https://github.com/yourusername/your-repo.git
git branch -M main
git push -u origin main
```

## Step 9: Deployment

### 9.1 Deploy to Render (or chosen platform)
1. Connect GitHub repository to Render
2. Configure build and start commands
3. Enable auto-deploy from GitHub
4. Monitor deployment status

### 9.2 Test Live Deployment
```bash
# Update live_post.py with your deployed URL
python live_post.py
```

## Step 10: Generate Required Screenshots

### 10.1 Required Screenshots/Files
- `example.png`: API documentation showing Pydantic example
- `live_get.png`: Browser screenshot of GET endpoint
- `continuous_integration.txt`: CI status screenshot
- `continuous_deployment.txt`: CD status screenshot
- `live_post_result.txt`: POST script results screenshot

## Expected Outputs

After completing all steps, you should have:

1. **Trained Model**: `model/model.pkl`, `model/encoder.pkl`, `model/lb.pkl`
2. **Performance Analysis**: `slice_output.txt` with metrics for all categorical feature slices
3. **Working API**: Local and deployed FastAPI application
4. **Test Results**: All tests passing (10+ tests total)
5. **Documentation**: Model card and project documentation
6. **CI/CD Pipeline**: Automated testing and deployment
7. **Screenshots**: All required visual evidence

## Verification Checklist

- [ ] Model trains successfully and saves artifacts
- [ ] Slice performance analysis generates comprehensive output
- [ ] All unit tests pass (pytest)
- [ ] Code style checks pass (flake8)
- [ ] API runs locally and responds correctly
- [ ] API documentation shows examples
- [ ] GitHub Actions CI passes
- [ ] Deployment succeeds
- [ ] Live API responds to requests
- [ ] All required screenshots captured

## Troubleshooting

### Common Issues:
1. **Import Errors**: Ensure `src/__init__.py` exists and virtual environment is activated
2. **Missing Dependencies**: Check `requirements.txt` and install missing packages
3. **Data Issues**: Verify census.csv format and handle missing values properly
4. **API Errors**: Check Pydantic model field aliases match CSV column names
5. **Deployment Issues**: Verify deployment configuration files and environment variables

### Performance Notes:
- Model training may take 1-2 minutes depending on hardware
- Slice performance analysis processes all categorical features (may take 2-3 minutes)
- API startup time is typically under 10 seconds

This reproduction guide ensures complete project replication following MLOps best practices and meeting all rubric requirements.

