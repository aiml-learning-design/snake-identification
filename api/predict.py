import cv2
import numpy as np
from tensorflow.keras.models import load_model
from src.image_handler import preprocess_image
from joblib import load

model = None
species_encoder = None


async def predict_snake_info(img_bytes: bytes):
    try:
        if model is None or species_encoder is None:
            load_resources()
        if img_bytes is None:
            return {"error": "Uploaded image is empty"}

        cv_image = read_image_bytes(img_bytes)

        preprocessed = preprocess_image(cv_image)
        input_tensor = np.expand_dims(preprocessed, axis=0)
        species_pred = model.predict(input_tensor)
        predicted_label = np.argmax(species_pred, axis=1)
        species_label = species_encoder.inverse_transform(predicted_label)[0]

        return {
            "species": species_label
        }

    except Exception as e:
        return {"error": str(e)}


def load_resources():
    global model, species_encoder
    model = load_model("models/snake_identifier_model.h5")
    species_encoder = load("models/species_encoder.pkl")


def read_image_bytes(img_bytes: bytes) -> np.ndarray:
    try:
        if not img_bytes:
            raise ValueError("Uploaded file is empty")
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Image decoding failed")
        cv_img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return cv_img_rgb

    except Exception as e:
        raise ValueError(f"Image processing failed: {str(e)}")



