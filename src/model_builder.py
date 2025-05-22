#from keras.src.preprocessing.image import ImageDataGenerator
from sklearn.utils import compute_class_weight
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras import layers, models, regularizers, metrics, optimizers, constraints
import tensorflow as tf
from tensorflow.keras.optimizers import AdamW


# In model_builder.py


def build_snake_model(num_species, num_venom_types, num_locations, num_toxicity):
    """Build the multi-output snake identification model"""

    base_model = EfficientNetV2B0(
        include_top=False,
        input_shape=(224, 224, 3),
        weights='imagenet'  # Fixed typo here
    )

    for layer in base_model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    inputs = layers.Input(shape=(224, 224, 3))

    x = base_model(inputs)

    # Add attention mechanism
    x = ChannelAttention(ratio=8)(x)
    x = SpatialAttention()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='swish',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    x = LayerNormalization()(x)

    # Output layers
    species_output = layers.Dense(num_species, activation='softmax', name='species_output',
                                  kernel_constraint=constraints.max_norm(2.0))(x)
    venom_output = layers.Dense(num_venom_types, activation='sigmoid', name='venom_output',
                                kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(x)
    location_output = layers.Dense(num_locations, activation='sigmoid', name='location_output')(x)
    toxicity_output = layers.Dense(num_toxicity, activation='sigmoid', name='toxicity_output')(x)

    # optimizer = optimizers.legacy.AdamW(
    #     learning_rate=0.0001,
    #     weight_decay=0.004
    # )

    optimizer = AdamW(
        learning_rate=0.0001,
        weight_decay=0.004
    )

    model = models.Model(inputs=inputs, outputs=[species_output, venom_output, location_output, toxicity_output])

    # Custom loss weights
    loss_weights = {
        "species_output": 1.0,
        "venom_output": 0.7,
        "location_output": 0.5,
        "toxicity_output": 0.5
    }

    model.compile(
        optimizer=optimizer,
        loss={
            "species_output": FocalLoss(gamma=2.0),
            "venom_output": "binary_crossentropy",
            "location_output": "binary_crossentropy",
            "toxicity_output": "binary_crossentropy"
        },
        metrics={
            "species_output": ["accuracy", metrics.SparseTopKCategoricalAccuracy(k=3)],
            "venom_output": ["accuracy", metrics.Precision(), metrics.Recall()],
            "location_output": "accuracy",
            "toxicity_output": "accuracy"
        },
        loss_weights=loss_weights
    )
    return model


class ChannelAttention(layers.Layer):
    def __init__(self, ratio=8, **kwargs):  # Add ratio parameter here
        super().__init__(**kwargs)
        self.ratio = ratio  # Store ratio as instance attribute

    def build(self, input_shape):
        channels = input_shape[-1]
        self.shared_dense = layers.Dense(channels // self.ratio,
                                         activation='relu',
                                         kernel_initializer='he_normal')
        self.output_dense = layers.Dense(channels,
                                         activation='sigmoid',
                                         kernel_initializer='he_normal')

    def call(self, inputs):
        # Channel attention mechanism
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)

        avg_out = self.output_dense(self.shared_dense(avg_pool))
        max_out = self.output_dense(self.shared_dense(max_pool))

        scale = tf.sigmoid(avg_out + max_out)
        return inputs * scale


def get_augmenter():
    return ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )



class SpatialAttention(layers.Layer):
    """Focuses on key spatial features (hood patterns, head shapes)"""
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(
            filters=1,
            kernel_size=kernel_size,
            padding='same',
            activation='sigmoid',
            kernel_initializer='he_normal'
        )

    def call(self, inputs):
        # Create attention map
        avg_out = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_out = tf.reduce_max(inputs, axis=-1, keepdims=True)
        attention = self.conv(tf.concat([avg_out, max_out], axis=-1))

        # Visualize attention (debug)
        if False:  # Set to True during debugging
            tf.print("Attention min/max:", tf.reduce_min(attention), tf.reduce_max(attention))

        return inputs * attention  # Apply attention weights

    def get_config(self):
        return super().get_config()


class FocalLoss(tf.keras.losses.Loss):
    """Handles class imbalance - critical for rare snake species"""
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        ce_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=False
        )
        pt = tf.exp(-ce_loss)  # prevents underflow
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return tf.reduce_mean(focal_loss)