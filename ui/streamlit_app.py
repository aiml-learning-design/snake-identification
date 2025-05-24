import io

import streamlit as st
import requests
from PIL import Image

API_URL = "http://localhost:7813/predict"

st.set_page_config(page_title="Snake Identifier", layout="centered")
st.title("Snake Identification Tool")

st.markdown("""
Upload an image of a snake, and the system will identify:
- Its **species**
- Possible **venom types**
- Likely **geographical regions**
""")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    uploaded_file.seek(0)
    img_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(img_bytes))
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Predicting..."):
        try:
            # Reset pointer and send file
            response = requests.post(
                API_URL,
                files={"file": (uploaded_file.name, img_bytes, uploaded_file.type)},
            )

            if response.status_code == 200:
                result = response.json()
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.subheader("Prediction Results")
                    st.markdown(f"**Species:** `{result['species']}`")

                    venom_types = result.get("venom_types", [])
                    regions = result.get("geographical_regions", [])

                    st.markdown("**Venom Types:**")
                    st.markdown(", ".join(venom_types) if venom_types else "No venom detected.")

                    st.markdown("**Likely Geographical Regions:**")
                    st.markdown(", ".join(regions) if regions else "Unknown region.")

            else:
                st.error(f"API Error: {response.status_code} - {response.text}")

        except Exception as e:
            st.error(f"Failed to process request: {str(e)}")