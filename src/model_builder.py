from tensorflow.keras import Sequential
from tensorflow.keras import Input
from tensorflow.keras.optimizers import legacy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, EfficientNetV2M
from tensorflow.keras import layers, models
import tensorflow as tf

# In model_builder.py

IMAGE_SIZE = 224


class ModelBuilder:

    def __init__(self, metadata_handler):
        self.metadata_handler = metadata_handler

    def build_conventional_model(self):
        model = models.Sequential([
            layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
            # Expanded Conv blocks
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),  # New layer
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),  # Regularization
            layers.Dense(len(self.metadata_handler.get_classes()), activation='softmax')
        ])
        model.compile(
            optimizer='adam',  # Use legacy Adam for M1/M2 stability
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def build_resnet50_model(self):
        # Load the pre-trained ResNet50 model without the top (classifier) layer
        base_model = ResNet50(input_shape=(224, 224, 3),
                              include_top=False,
                              weights='imagenet')

        # Freeze the base model (so we only train the top layers)
        base_model.trainable = False

        # Build the custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(self.metadata_handler.get_classes()), activation='softmax')
        ])

        # Compile the model
        model.compile(
            optimizer=AdamW(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def build_snake_model(self):
        """Build the multi-output snake identification model"""

        base_model = MobileNetV2(
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False

        for layer in base_model.layers[:150]:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)

        #  x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
        # x = BatchNormalization()(x)

        predictions = layers.Dense(
            len(self.metadata_handler.get_classes()),
            activation='softmax',
            kernel_regularizer=l2(0.01),
            kernel_initializer='glorot_uniform'
        )(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

        model.compile(
            optimizer=AdamW(learning_rate=1e-4),
            # loss='categorical_crossentropy',
            loss=loss_fn,
            metrics=['accuracy']
        )
        return model

    def build_dnn(self, X_train):
        input_shape = (X_train.shape[1],)
        model = Sequential([
            # Input Dense Block
            Dense(512, activation='relu', input_shape=input_shape, kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Dropout(0.4),

            # Intermediate Dense Block
            Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Dropout(0.3),

            # Light Dense Block
            Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Dropout(0.2),

            # Output
            Dense(len(self.metadata_handler.get_classes()), activation='softmax')
        ])

        optimizer = AdamW(learning_rate=0.0003)  # slightly lower lr for better convergence
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def build_best_snake_model(image_size=384):
        base_model = EfficientNetV2M(
            input_shape=(image_size, image_size, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )

        # Freeze initial layers
        base_model.trainable = True
        for layer in base_model.layers[:100]:
            layer.trainable = False

        inputs = Input(shape=(image_size, image_size, 3))
        x = base_model(inputs)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='swish')(x)
        outputs = Dense(13, activation='softmax')(x)

        model = Model(inputs, outputs)

        model.compile(
            optimizer=legacy.AdamW(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model


def get_data_generators(data_dir, batch_size=32):
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=30,
        brightness_range=[0.7,1.5],
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2,
        fill_mode='nearest'
    )

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_gen, val_gen
