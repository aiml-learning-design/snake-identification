from http.client import HTTPException

import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import joblib
from fastapi import UploadFile
from tensorflow.keras.models import load_model
from src.image_handler import preprocess_image

from fastapi import HTTPException
from fastapi import status  # For HTTP status codes

model = None
species_encoder = None
venom_encoder = None
location_encoder = None


def load_resources():
    global model, species_encoder, venom_encoder, location_encoder
    model = load_model("models/snake_identifier_model")
    species_encoder = joblib.load("models/species_encoder.pkl")
    venom_encoder = joblib.load("models/venom_encoder.pkl")
    location_encoder = joblib.load("models/location_encoder.pkl")


async def read_image(uploaded_file: UploadFile) -> np.ndarray:
    try:
        # Reset file pointer (critical!)
        await uploaded_file.seek(0)

        # Read file content
        img_bytes = await uploaded_file.read()

        if not img_bytes:
            raise ValueError("Uploaded file is empty")

        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return np.array(pil_img)  # Shape: (H, W, 3)
    except UnidentifiedImageError:
        raise ValueError("Invalid image format")
    except Exception as e:
        raise ValueError(f"Image processing failed: {str(e)}")


async def predict_snake_info(image_file: UploadFile):
    try:
        if model is None:
            load_resources()

        img = await read_image(image_file)

        img_array = np.array(img)
        preprocessed = preprocess_image(img_array)
        input_tensor = np.expand_dims(preprocessed, axis=0)

        species_pred, venom_pred, location_pred = model.predict(input_tensor)

        # Decode predictions
        species_label = species_encoder.inverse_transform([np.argmax(species_pred)])[0]
        venom_labels = venom_encoder.inverse_transform((venom_pred[0] > 0.5).astype(int).reshape(1, -1))[0]
        location_labels = location_encoder.inverse_transform((location_pred[0] > 0.5).astype(int).reshape(1, -1))[0]

        return {
            "species": species_label,
            "venom_types": [v for v in venom_labels if v],
            "geographical_regions": [l for l in location_labels if l]
        }

    except Exception as e:
        return {"error": str(e)}
