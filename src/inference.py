import joblib
#from keras.src.saving import load_model
#from tensorflow.keras.models import load_model
#import tensorflow as tf

# Option 1: Modern import
from tensorflow.keras.saving import load_model

# Option 2: Legacy import
#from tensorflow.keras.models import load_model



model = None
species_encoder = None
venom_encoder = None
location_encoder = None


def load_resources():
    global model, species_encoder, venom_encoder, location_encoder
    if model is None:
        model = load_model("models/snake_identifier_model")
        species_encoder = joblib.load("models/species_encoder.pkl")
        venom_encoder = joblib.load("models/venom_encoder.pkl")
        location_encoder = joblib.load("models/location_encoder.pkl")
