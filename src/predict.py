import io
import joblib
import numpy as np
from PIL import Image

MODEL_PATH = "model/digit_model.pkl"

# Load trained digit model once when the app starts
model = joblib.load(MODEL_PATH)


def preprocess_image(image_bytes):
    """
    Convert uploaded image bytes into the 8x8 grayscale format
    expected by sklearn's digits dataset.
    """

    image = Image.open(io.BytesIO(image_bytes))

    # Convert to grayscale
    image = image.convert("L")

    # Resize to 8x8
    image = image.resize((8, 8))

    # Convert to numpy array
    image_array = np.array(image, dtype=np.float32)

    # The sklearn digits dataset uses:
    # background ~ 0
    # digit strokes ~ 16
    #
    # Typical uploaded handwritten images often use:
    # background ~ 255
    # digit strokes ~ 0
    #
    # Invert and scale from 0-255 to 0-16.
    image_array = 255.0 - image_array
    image_array = (image_array / 255.0) * 16.0

    # Flatten 8x8 into 64 features
    features = image_array.reshape(1, -1)

    return features


def predict_digit(image_bytes):
    """
    Predict handwritten digit and confidence score.
    """

    features = preprocess_image(image_bytes)

    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]
    confidence = float(np.max(probabilities))

    return int(prediction), confidence


if __name__ == "__main__":
    print("Digit model loaded successfully")