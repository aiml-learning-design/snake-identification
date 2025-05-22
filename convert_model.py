from tensorflow.keras.layers import TFSMLayer
from tensorflow.keras.models import save_model
import tensorflow as tf

# Load the SavedModel as a layer
model_layer = TFSMLayer("models/snake_identifier_model",
                        call_endpoint='serving_default')

# Build a new model with this layer
inputs = tf.keras.Input(shape=(224, 224, 3))  # Adjust shape to your model's input
outputs = model_layer(inputs)
new_model = tf.keras.Model(inputs, outputs)

# Save in new format
new_model.save("models/converted_model.keras")  # New recommended format
new_model.save("models/converted_model.h5")     # Legacy H5 format