import streamlit as st
import requests
from PIL import Image

API_URL = "https://handwritten-digit-api-916630904861.us-east1.run.app"

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢"
)

st.title("Handwritten Digit Recognition")

st.write(
    "Upload an image containing a handwritten digit from 0 to 9."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded image",
        width=250
    )

    if st.button("Predict Digit"):
        files = {
            "image": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        try:
            response = requests.post(
                f"{API_URL}/predict",
                files=files,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()

                prediction = result["prediction"]
                confidence = result["confidence"]

                st.success(f"Predicted digit: {prediction}")

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.1f}%"
                )

            else:
                st.error(
                    f"API error: {response.text}"
                )

        except requests.exceptions.RequestException as exc:
            st.error(
                f"Could not connect to prediction API: {exc}"
            )