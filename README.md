# Handwritten Digit Recognition API with Flask, Docker, and Google Cloud Run

## Project Overview

This project is a modified version of the original Flask GCP lab.

Instead of serving an Iris flower classifier, this project implements a handwritten digit recognition system that accepts an image of a digit from 0 to 9, preprocesses the image, sends it through a trained machine learning model, and returns the predicted digit along with a confidence score.

The project also extends the original lab by adding API validation, health and model information endpoints, automated API tests, Docker health monitoring, and deployment to Google Cloud Run.

## Architecture

The end to end workflow is:

Handwritten digit image
        |
        v
Streamlit frontend
        |
        v
Flask API
        |
        v
Image preprocessing
        |
        v
Random Forest model
        |
        v
Prediction + confidence
        |
        v
JSON response

The Flask API is containerized using Docker and deployed to Google Cloud Run.

## Main Components

### Model Training

The model is trained using the scikit-learn handwritten digits dataset.

Dataset: sklearn.datasets.load_digits

Number of samples: 1797

Number of classes: 10

Input features: 64

Model: RandomForestClassifier

Test accuracy achieved: approximately 96.4%

The trained model is saved as:

model/digit_model.pkl

### API Endpoints

#### Health Check

GET /health

Example response:

{
  "service": "handwritten-digit-classifier",
  "status": "healthy"
}

#### Model Information

GET /model-info

Example response:

{
  "model": "RandomForestClassifier",
  "dataset": "scikit-learn digits",
  "classes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  "input_format": "image file",
  "image_size": "8x8 grayscale after preprocessing"
}

#### Prediction

POST /predict

The API expects an image in the multipart form field:

image

Supported file types:

PNG

JPG

JPEG

Example response:

{
  "prediction": 7,
  "confidence": 0.35
}

## Project Structure

mlops-lab2/
|
|-- model/
|   |-- digit_model.pkl
|
|-- src/
|   |-- train.py
|   |-- predict.py
|   |-- main.py
|   |-- test_api.py
|
|-- Dockerfile
|-- requirements.txt
|-- streamlit_app.py
|-- digit7.png
|-- README.md

## Prerequisites

The following tools are required.

### Python

Recommended:

Python 3.11+

Check:

python --version

### Git

Check:

git --version

### Docker Desktop

Install Docker Desktop for your operating system.

Check:

docker --version

Make sure Docker Desktop is running before building or running containers.

### Google Cloud CLI

Install the Google Cloud CLI.

Check:

gcloud --version

## Environment Setup

Clone the repository:

git clone https://github.com/Sivapriya2123/mlops-flask-api-gcp.git

Move into the project:

cd mlops-flask-api-gcp

Create a virtual environment:

python -m venv venv

Activate it on macOS or Linux:

source venv/bin/activate

On Windows:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

## Model Training

Run:

python src/train.py

Expected output is similar to:

Dataset shape: (1797, 64)
Number of classes: 10
Model accuracy: 0.9639
Saved trained model to model/digit_model.pkl

Confirm that the model exists:

ls -lh model

## Run Flask API Locally

From the project root:

python src/main.py

The API should start on:

http://127.0.0.1:8080

## Test the API Locally

### Health Endpoint

curl http://127.0.0.1:8080/health

### Model Information

curl http://127.0.0.1:8080/model-info

### Prediction

Example:

curl -X POST \
  -F "image=@digit7.png" \
  http://127.0.0.1:8080/predict

Example output:

{
  "confidence": 0.35,
  "prediction": 7
}

## Automated Testing

The project includes automated Flask API tests.

The tests cover:

health endpoint

model information endpoint

valid prediction

missing image

unsupported file type

corrupted image

Run:

PYTHONPATH=src python -m pytest src/test_api.py -v

Expected result:

6 passed

Example successful test run:

test_health_endpoint PASSED
test_model_info_endpoint PASSED
test_predict_missing_image PASSED
test_predict_invalid_file_type PASSED
test_predict_corrupted_image PASSED
test_valid_prediction PASSED

## Docker

### Build Docker Image

Run:

docker build -t handwritten-digit-api .

### Run Container

docker run -p 8080:8080 handwritten-digit-api

The container exposes the API on:

http://127.0.0.1:8080

### Docker Health Check

The Dockerfile includes a health check that periodically calls:

GET /health

This verifies that the Flask service is running correctly inside the container.

## Google Cloud Setup

Authenticate with Google Cloud:

gcloud auth login

Initialize the CLI:

gcloud init

Set the project:

gcloud config set project nice-champion-511522-h6

Confirm:

gcloud config get-value project

### Enable Required APIs

Run:

gcloud services enable cloudbuild.googleapis.com

gcloud services enable run.googleapis.com

gcloud services enable artifactregistry.googleapis.com

Confirm:

gcloud services list --enabled --filter="NAME:(cloudbuild.googleapis.com OR run.googleapis.com OR artifactregistry.googleapis.com)"

## Build Container in Google Cloud

Run:

gcloud builds submit --tag gcr.io/nice-champion-511522-h6/handwritten-digit-api

A successful build should end with:

STATUS: SUCCESS

## Deploy to Google Cloud Run

Run:

```
gcloud run deploy handwritten-digit-api \
  --image gcr.io/nice-champion-511522-h6/handwritten-digit-api \
  --platform managed \
  --region us-east1 \
  --allow-unauthenticated \
  --port 8080
```

After deployment, Google Cloud will provide a service URL.

For this project, the deployed API was tested successfully using:

https://handwritten-digit-api-916630904861.us-east1.run.app

## Test the Deployed Cloud Run API

### Health

curl https://handwritten-digit-api-916630904861.us-east1.run.app/health

Expected response:

{
  "service": "handwritten-digit-classifier",
  "status": "healthy"
}

### Model Information

curl https://handwritten-digit-api-916630904861.us-east1.run.app/model-info

### Prediction

curl -X POST \
  -F "image=@digit7.png" \
  https://handwritten-digit-api-916630904861.us-east1.run.app/predict

Example result:

{
  "confidence": 0.35,
  "prediction": 7
}

## Streamlit Frontend

The project includes a Streamlit frontend where users can upload a handwritten digit image.

Run:

streamlit run streamlit_app.py

The browser will open at:

http://localhost:8501

Upload an image and click:

Predict Digit

The application displays:

Predicted digit: 7

Confidence: 35.0%

## Meaningful Changes Made to the Original Lab

This project does more than replace the original Iris dataset or model.

The original Flask GCP lab accepted four numeric Iris measurements and returned a flower class.

This implementation modifies the complete model serving workflow.

The major changes include:

1. Replaced numeric Iris input with image based handwritten digit input.

2. Added image preprocessing including grayscale conversion, resizing to 8x8, pixel inversion, normalization, and flattening to model features.

3. Added prediction confidence using predict_proba.

4. Added a GET /health endpoint.

5. Added a GET /model-info endpoint.

6. Added API input validation.

7. Added proper HTTP 400 error responses.

8. Added validation for unsupported file types.

9. Added handling for corrupted image files.

10. Added multiple automated API tests.

11. Added Docker health monitoring.

12. Updated the container setup.

13. Deployed the modified API to Google Cloud Run.

14. Updated the Streamlit frontend to upload images and call the deployed Cloud Run API.

Therefore, the modification affects not only the dataset and model, but also the API contract, preprocessing pipeline, testing pipeline, container behavior, health monitoring, and cloud deployment workflow.

## Successful End to End Workflow

The completed workflow is:

User uploads handwritten digit
        |
        v
Streamlit frontend
        |
        v
Google Cloud Run
        |
        v
Flask API
        |
        v
Input validation
        |
        v
Image preprocessing
        |
        v
Random Forest model
        |
        v
Prediction + confidence
        |
        v
JSON response
        |
        v
Streamlit displays result

The complete system was tested successfully both locally and after deployment to Google Cloud Run.
