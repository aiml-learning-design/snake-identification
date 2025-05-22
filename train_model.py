import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import train_test_split
import joblib
import os
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.data_preparation import DatasetBuilder
from src.model_builder import build_snake_model
from src.metadata_handler import MetadataHandler

import seaborn as sns

os.makedirs("models", exist_ok=True)


def main():
    # Initialize metadata handler and dataset builder
    metadata_handler = MetadataHandler("data/image_metadata.csv")
    print(f"Total images in metadata: {len(metadata_handler.metadata)}")
    print(f"Unique species in metadata: {metadata_handler.metadata['species'].nunique()}")
    builder = DatasetBuilder("data/raw_images", metadata_handler)

    # Load and prepare data
    try:
        X, y_species, y_venom, y_geo, y_toxicity = builder.load_image_and_labels()

        print(f"NaN in species labels: {np.isnan(y_species).any()}")
        print(f"NaN in venom labels: {np.isnan(y_venom).any()}")

        print(f"First 5 species labels: {y_species[:5]}")
        print(f"First 5 venom labels: {y_venom[:5]}")

        print(f"Loaded {len(X)} images with shapes:")
        print(f"Species labels: {y_species.shape}")
        print(f"Venom labels: {y_venom.shape}")
        print(f"Geo labels: {y_geo.shape}")
        print(f"Toxicity labels: {y_toxicity.shape}")

        if len(X) == 0:
            raise ValueError("No images were loaded!")
        if len(y_species) == 0:
            raise ValueError("No species labels were loaded!")
        if len(X) != len(y_species):
            raise ValueError(f"Mismatch between images ({len(X)}) and labels ({len(y_species)})")
    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        return

    # Split data with stratification
    X_train, X_val, y_species_train, y_species_val, y_venom_train, y_venom_val, y_geo_train, y_geo_val, y_toxicity_train, y_toxicity_val = train_test_split(
        X, y_species, y_venom, y_geo, y_toxicity,
        test_size=0.2,
        random_state=42,
        stratify=y_species
    )

    # Convert images to float32 and normalize
    X_train = np.array([img.astype('float32') / 255.0 for img in X_train])
    X_val = np.array([img.astype('float32') / 255.0 for img in X_val])

    # Verify shapes after split
    print(f"\nAfter split:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_species_train shape: {y_species_train.shape}")
    print(f"y_venom_train shape: {y_venom_train.shape}")
    print(f"y_geo_train shape: {y_geo_train.shape}")
    print(f"y_toxicity_train shape: {y_toxicity_train.shape}")

    # Initialize data augmenter
    train_datagen = get_snake_augmenter()

    # Class-aware sample weighting
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_species),
        y=y_species
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

    # Boost problematic classes (King Cobra, etc.)
    problem_classes = ["Ophiophagus hannah", "Argyrogena fasciolata"]
    for i, cls in enumerate(builder.label_encoder.classes_):
        print(f"{cls}: {class_weight_dict.get(i, 1.0)}")
        if cls in problem_classes:
            class_weights[i] *= 2.5  # Stronger weighting

    sample_weights = np.array([class_weights[i] for i in y_species_train])

    # Create TensorBoard log directory
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    # Enhanced callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_species_output_sparse_top_k_categorical_accuracy',
            patience=15,
            mode='max',
            restore_best_weights=True
        ),
        ModelCheckpoint(
            "models/best_model.h5",
            monitor='val_species_output_recall',
            mode='max',
            save_best_only=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )
    ]

    # y_species_train = np.array(y_species_train)
    # y_venom_train = np.array(y_venom_train)
    # y_geo_train = np.array(y_geo_train)
    # y_toxicity_train = np.array(y_toxicity_train)

    # Create data generator
    # train_generator = train_datagen.flow(
    #     X_train,
    #     {
    #         'species_output': y_species_train,
    #         'venom_output': y_venom_train,
    #         'location_output': y_geo_train,
    #         'toxicity_output': y_toxicity_train
    #      },
    #     batch_size=32
    #   # , sample_weight=sample_weights
    # )

    # y_species_val = np.array(y_species_val)
    # y_venom_val = np.array(y_venom_val)
    # y_geo_val = np.array(y_geo_val)
    # y_toxicity_val = np.array(y_toxicity_val)

    # train_generator = train_datagen.flow(
    #     X_train,
    #     {
    #         'species_output': y_species_train,
    #         'venom_output': y_venom_train,
    #         'location_output': y_geo_train,
    #         'toxicity_output': y_toxicity_train
    #     },
    #     batch_size=32
    # )

    batch_size = 32

    train_generator = multi_output_generator(
        X_train, y_species_train, y_venom_train, y_geo_train, y_toxicity_train, sample_weights, batch_size, train_datagen
    )
    steps_per_epoch = len(X_train)

    sample_weight = {
        'species_output': sample_weights,
        'venom_output': np.ones(len(y_venom_train)),
        'location_output': np.ones(len(y_geo_train)),
        'toxicity_output': np.ones(len(y_toxicity_train))
    }

    # Build model
    try:
        model = build_snake_model(
            num_species=len(builder.label_encoder.classes_),
            num_venom_types=y_venom.shape[1],
            num_locations=y_geo.shape[1],
            num_toxicity=y_toxicity.shape[1]
        )
    except Exception as e:
        print(f"Model building failed: {str(e)}")
        return

    # Train model
    # history = model.fit(
    #     train_generator,
    #     steps_per_epoch=len(X_train) // 32,
    #     validation_data=(X_val, {
    #         "species_output": y_species_val,
    #         "venom_output": y_venom_val,
    #         "location_output": y_geo_val,
    #         "toxicity_output": y_toxicity_val
    #     }),
    #     epochs=100,
    #     callbacks=callbacks,
    #     class_weight={'species_output': dict(enumerate(class_weights))},
    #     verbose=1
    # )

    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // 32,
        validation_data=(X_val, {
            "species_output": y_species_val,
            "venom_output": y_venom_val,
            "location_output": y_geo_val,
            "toxicity_output": y_toxicity_val
        }),
        epochs=100,
        callbacks=callbacks,
        # class_weight={
        #     'species_output': class_weight_dict,
        #     'venom_output': None,  # No class weights for other outputs
        #     'location_output': None,
        #     'toxicity_output': None
        # },
        verbose=1
    )

    # Save models and generate reports
    model.save("models/snake_identifier_model")
    joblib.dump(builder.label_encoder, "models/species_encoder.pkl")
    joblib.dump(builder.venom_encoder, "models/venom_encoder.pkl")
    joblib.dump(builder.location_encoder, "models/location_encoder.pkl")
    joblib.dump(builder.toxicity_encoder, "models/toxicity_encoder.pkl")

    generate_evaluation_report(model, X_val, y_species_val, builder.label_encoder)
    print("Training completed and models saved!")


def generate_evaluation_report(model, X_val, y_species_val, label_encoder):
    y_pred = model.predict(X_val)
    y_pred_species = np.argmax(y_pred[0], axis=1)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_species_val, y_pred_species,
                                target_names=label_encoder.classes_))

    # Confusion matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_species_val, y_pred_species)
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.savefig('models/confusion_matrix.png')
    plt.close()


def multi_output_generator(X, y_species, y_venom, y_geo, y_toxicity, sample_weights, batch_size, augmenter):
    sample_count = len(X)
    indices = np.arange(sample_count)

    while True:
        np.random.shuffle(indices)

        for start in range(0, sample_count, batch_size):
            end = min(start + batch_size, sample_count)
            batch_ids = indices[start:end]

            batch_X = X[batch_ids]
            batch_y_species = y_species[batch_ids]
            batch_y_venom = y_venom[batch_ids]
            batch_y_geo = y_geo[batch_ids]
            batch_y_toxicity = y_toxicity[batch_ids]
            batch_weights = sample_weights[batch_ids]

            # Apply augmentation
            batch_augmented = np.zeros_like(batch_X)
            for i in range(len(batch_X)):
                batch_augmented[i] = augmenter.random_transform(batch_X[i])

            yield batch_augmented, {
                'species_output': batch_y_species,
                'venom_output': batch_y_venom,
                'location_output': batch_y_geo,
                'toxicity_output': batch_y_toxicity
            }, {
                'species_output': batch_weights,
                'venom_output': np.ones(len(batch_ids)),
                'location_output': np.ones(len(batch_ids)),
                'toxicity_output': np.ones(len(batch_ids))
            }


def get_snake_augmenter():
    """Specialized augmentation for snake images"""
    return ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='reflect',
        brightness_range=[0.9, 1.1],
        channel_shift_range=10.0
    )


if __name__ == "__main__":
    main()
