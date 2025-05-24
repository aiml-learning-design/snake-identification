from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras import Model

# In model_builder.py

IMAGE_SIZE = 224


class ModelBuilder:

    def __init__(self, metadata_handler):
        self.metadata_handler = metadata_handler

    def build_snake_model(self, y):
        """Build the multi-output snake identification model"""

        base_model = MobileNetV2(
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)

        predictions = Dense(len(self.metadata_handler.get_classes()), activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(
            optimizer=AdamW(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model