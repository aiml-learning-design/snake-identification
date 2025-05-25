from tensorflow.keras.models import load_model
print("Loading model...")
model = load_model("models/snake_identifier_model.keras")
print("Model loaded successfully.")