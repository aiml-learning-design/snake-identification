import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from src.image_handler import preprocess_image
from src.metadata_handler import MetadataHandler
from tensorflow.keras.models import load_model
import joblib



# Full-width layout
st.set_page_config(page_title="Snake Identifier", layout="wide")

# Custom Header with full width and styles
st.markdown("""
<style>
[data-testid="stAppViewContainer"] > .main {
    padding-left: 0rem;
    padding-right: 0rem;
}

.block-container {
    padding: 0rem 2rem 2rem 2rem;
    max-width: 100% !important;
}

.header-container {
    width: 100%;
    background-color: #1e1e1e;
    padding: 30px;
    border-radius: 12px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    text-align: center;
    margin-bottom: 25px;
}
.header-container h1 {
    color: #f1c40f;
    font-size: 52px;
    margin-bottom: 10px;
    font-weight: bold;
    letter-spacing: 2px;
}
.header-container p {
    color: #ecf0f1;
    font-size: 20px;
    margin-top: 0;
}
</style>

<div class="header-container">
    <h1>Snake Identification Tool 🐍</h1>
    <p>Upload a snake image and instantly identify its species, venom level, habitat, and more!</p>
</div>
""", unsafe_allow_html=True)

# Load resources once
@st.cache_resource
def load_resources():
    model = load_model("snake_identifier_model.keras")
    species_encoder = joblib.load("specie_encoder.pkl")
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
<style>
.snake-info-container {
    display: flex;
    justify-content: space-between;
    margin-top: 20px;
    gap: 3%;
}
.snake-column {
    flex: 1;
    padding: 20px;
    border-radius: 12px;
    background-color: #f9f9f9;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}
.snake-column h3 {
    text-align: center;
    color: white;
    padding: 10px;
    border-radius: 8px;
}
.non-venomous h3 {
    background-color: #4CAF50; /* Green */
}
.venomous h3 {
    background-color: #f44336; /* Red */
}
.snake-column ul {
    padding-left: 20px;
    font-size: 16px;
}
.heading-block {
    background-color: #2c3e50;
    color: white;
    padding: 16px;
    border-radius: 10px;
    margin-bottom: 10px;
    text-align: center;
}
</style>

<div class="heading-block">
    <h2>Supported Snake Categories</h2>
    <p>Below is the list of Indian snakes currently supported by the identification system, grouped by venom type.</p>
</div>

<div class="snake-info-container">
    <div class="snake-column non-venomous">
        <h3>🟢 Non-Venomous</h3>
        <ul>
            <li>Banded Racer</li>
            <li>Checkered Keelback</li>
            <li>Common Rat Snake</li>
            <li>Common Sand Boa</li>
            <li>Common Trinket</li>
            <li>Green Tree Vine</li>
            <li>Indian Rock Python</li>
        </ul>
    </div>
    <div class="snake-column venomous">
        <h3>🔴 Venomous</h3>
        <ul>
            <li>Common Krait</li>
            <li>King Cobra</li>
            <li>Monocled Cobra</li>
            <li>Russell's Viper</li>
            <li>Saw-scaled Viper</li>
            <li>Spectacled Cobra</li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)


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
