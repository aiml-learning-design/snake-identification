import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from src.image_handler import preprocess_image
from src.metadata_handler import MetadataHandler
from tensorflow.keras.models import load_model
import joblib


# Streamlit UI
st.set_page_config(page_title="Snake Identifier", layout="centered")
st.title("Snake Identification Tool")

# Load resources once
@st.cache_resource
def load_resources():
    model = load_model("snake_identifier_model.keras")
    species_encoder = joblib.load("species_encoder.pkl")
    metadata_handler = MetadataHandler("image_metadata.csv")
    return model, species_encoder, metadata_handler

model, species_encoder, metadata_handler = load_resources()

# Prediction function
def predict_snake_info(img_bytes: bytes):
    try:
        if img_bytes is None:
            return {"error": "Uploaded image is empty"}

        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return {"error": "Image decoding failed"}
        cv_img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        preprocessed = preprocess_image(cv_img_rgb)
        input_tensor = np.expand_dims(preprocessed, axis=0)
        species_pred = model.predict(input_tensor)
        predicted_label = np.argmax(species_pred, axis=1)
        species_label = species_encoder.inverse_transform(predicted_label)[0]

        return {
            "species": species_label,
            "CommonName": metadata_handler.get_common_name(species_label),
            "Venom": metadata_handler.get_venom_type(species_label),
            "Toxicity": metadata_handler.get_toxicity_level(species_label),
            "Anti Venom Available": metadata_handler.is_anti_venom_available(species_label),
            "Location": metadata_handler.get_geo_info(species_label),
            "Habitat": metadata_handler.get_habitat_info(species_label)
        }

    except Exception as e:
        return {"error": str(e)}



st.markdown("""
Upload an image of a snake, and the system will identify:
- Its **species**
- Possible **venom types**
- Likely **geographical regions**
""")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"])

if uploaded_file is not None:
    uploaded_file.seek(0)
    img_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(img_bytes))
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Predicting..."):
        result = predict_snake_info(img_bytes)

        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            st.subheader("Prediction Results")
            st.markdown(f"**Species:** `{result['species']}`")
            st.markdown(f"**Common Name:** `{result['CommonName']}`")
            st.markdown(f"**Venom Types:** `{result['Venom']}`")
            st.markdown(f"**Anti Venom Available:** `{result['Anti Venom Available']}`")
            st.markdown(f"**Toxicity Level:** `{result['Toxicity']}`")
            st.markdown(f"**Likely Geographical Regions:** `{result['Location']}`")
            st.markdown(f"**Habitat Environment:** `{result['Habitat']}`")
