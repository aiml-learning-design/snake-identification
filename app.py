import base64
import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from src.image_handler import preprocess_image
from src.metadata_handler import MetadataHandler
from tensorflow.keras.models import load_model
import joblib
import os
from datetime import datetime

# Full-width layout
st.set_page_config(page_title="Snake Identifier", layout="wide")

# Main container styling
st.markdown("""
<style>
    /* Main container adjustments */
    [data-testid="stAppViewContainer"] > .main {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* Tab container styling */
    [data-testid="stHorizontalBlock"] {
        gap: 0.5rem;
    }
    
    /* Header styling */
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
    
    /* Tab content padding */
    .stTabs [data-baseweb="tab-panel"] {
        padding: 1rem 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Create tabs FIRST before any other content
# Inject custom CSS for tab styling
# Inject CSS to style the tab labels


# Custom CSS for tabs
st.markdown("""
<style>
    /* Targets ALL tab labels */
    div[data-testid="stTabs"] button div p {
        font-size: 20px !important;
        font-weight: 900 !important;
    }
    
    /* Active tab highlight */
    div[data-testid="stTabs"] button[aria-selected="true"] {
        background-color: #f0f2f6 !important;
        border-bottom: 3px solid #ff4b4b !important;
    }
</style>
""", unsafe_allow_html=True)

# Create tabs with larger bold text
tab1, tab2 = st.tabs(["📷 Identify Snake", "📤 Incorrect Prediction?"])
with tab1:
    # Header content
    st.markdown("""
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

    # Snake info columns
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

    # Image upload and prediction
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"])

    if uploaded_file is not None:
        uploaded_file.seek(0)
        img_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(img_bytes))

        # Display scaled image
        image_base64 = base64.b64encode(img_bytes).decode()

        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; align-items: center; margin: 20px 0;">
                <img src="data:image/png;base64,{image_base64}" 
                     style="width: 50%; height: auto; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);" 
                     alt="Uploaded Snake Image">
            </div>
            """,
            unsafe_allow_html=True
        )

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

with tab2:
    st.markdown("""
    <div class="header-container" style="background-color: #2c3e50;">
        <h1>Help Improve Our Model 🛠️</h1>
        <p>Your corrections help make our snake identification system more accurate!</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("If the snake was incorrectly identified, please provide the correct information below:")

    col1, col2 = st.columns(2)

    with col1:
        correction_file = st.file_uploader(
            "Upload the image again",
            type=["jpg", "jpeg", "png"],
            key="correction"
        )

    with col2:
        correct_species = st.text_input(
            "What is the correct snake species?",
            placeholder="e.g., 'King Cobra'"
        )
        venomous_type = st.radio(
            "Venom type",
            ["Venomous", "Non-Venomous"],
            horizontal=True
        )

    if st.button("Submit Correction", type="primary"):
        if correction_file and correct_species:
            try:
                # Create folder if not exists
                folder = f"data/corrections/{venomous_type.lower().replace('-', '')}"
                os.makedirs(folder, exist_ok=True)

                # Save file with timestamp and correct name
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                file_ext = os.path.splitext(correction_file.name)[-1]
                filename = f"{correct_species.replace(' ', '_')}_{timestamp}{file_ext}"
                filepath = os.path.join(folder, filename)

                # Write to file
                with open(filepath, "wb") as f:
                    f.write(correction_file.getbuffer())

                st.success(f"✅ Thank you! Your correction has been recorded.")
            except Exception as e:
                st.error(f"❌ Error saving your correction: {str(e)}")
        else:
            st.warning("⚠️ Please upload an image and specify the correct species name")


st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 5px;
        font-size: 0.8em;
        color: #6c757d;  /* Gray for subtlety */
        background-color: rgba(255, 255, 255, 0.5);  /* Semi-transparent white */
        border-top: 1px solid #e9ecef;  /* Thin border */
    }
    </style>
    <div class="footer">
        Developed by <strong>DHEERAJ KUMAR</strong>
    </div>
    """,
    unsafe_allow_html=True
)