import io

import pytest
from PIL import Image

from main import app


@pytest.fixture
def client():
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client


def create_test_image():
    image = Image.new("L", (28, 28), color=255)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    return buffer


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.get_json()

    assert data["status"] == "healthy"
    assert data["service"] == "handwritten-digit-classifier"


def test_model_info_endpoint(client):
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.get_json()

    assert data["model"] == "RandomForestClassifier"
    assert data["dataset"] == "scikit-learn digits"
    assert data["classes"] == list(range(10))


def test_predict_missing_image(client):
    response = client.post("/predict")

    assert response.status_code == 400

    data = response.get_json()

    assert data["error"] == "No image file provided"


def test_predict_invalid_file_type(client):
    response = client.post(
        "/predict",
        data={
            "image": (
                io.BytesIO(b"not an image"),
                "sample.txt"
            )
        },
        content_type="multipart/form-data"
    )

    assert response.status_code == 400

    data = response.get_json()

    assert data["error"] == "Unsupported file type"


def test_predict_corrupted_image(client):
    response = client.post(
        "/predict",
        data={
            "image": (
                io.BytesIO(b"this is not a real png image"),
                "sample.png"
            )
        },
        content_type="multipart/form-data"
    )

    assert response.status_code == 400

    data = response.get_json()

    assert data["error"] == "Unable to process image"


def test_valid_prediction(client):
    image = create_test_image()

    response = client.post(
        "/predict",
        data={
            "image": (
                image,
                "digit.png"
            )
        },
        content_type="multipart/form-data"
    )

    assert response.status_code == 200

    data = response.get_json()

    assert "prediction" in data
    assert "confidence" in data

    assert 0 <= data["prediction"] <= 9
    assert 0.0 <= data["confidence"] <= 1.0