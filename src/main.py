from flask import Flask, request, jsonify
from predict import predict_digit
import os

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Welcome to the Handwritten Digit Recognition API!",
        "description": "This Flask API predicts handwritten digits from 0 to 9 using a Random Forest classifier.",
        "model": "RandomForestClassifier",
        "dataset": "scikit-learn Digits",
        "endpoints": {
            "GET /": "API homepage",
            "GET /health": "Check API health",
            "GET /model-info": "View model information",
            "POST /predict": "Upload a digit image using the image form field"
        },
        "github": "https://github.com/Sivapriya2123/mlops-flask-api-gcp"
    }), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "handwritten-digit-classifier"
    }), 200


@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "model": "RandomForestClassifier",
        "dataset": "scikit-learn digits",
        "classes": list(range(10)),
        "input_format": "image file",
        "image_size": "8x8 grayscale after preprocessing"
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    # Check whether an image was included
    if "image" not in request.files:
        return jsonify({
            "error": "No image file provided",
            "expected_field": "image"
        }), 400

    image_file = request.files["image"]

    # Check for empty filename
    if image_file.filename == "":
        return jsonify({
            "error": "No image selected"
        }), 400

    # Restrict supported file types
    allowed_extensions = {"png", "jpg", "jpeg"}

    if "." not in image_file.filename:
        return jsonify({
            "error": "Invalid file type"
        }), 400

    extension = image_file.filename.rsplit(".", 1)[1].lower()

    if extension not in allowed_extensions:
        return jsonify({
            "error": "Unsupported file type",
            "supported_types": sorted(allowed_extensions)
        }), 400

    try:
        image_bytes = image_file.read()

        prediction, confidence = predict_digit(image_bytes)

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 4)
        }), 200

    except Exception as exc:
        return jsonify({
            "error": "Unable to process image",
            "details": str(exc)
        }), 400


if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080))
    )