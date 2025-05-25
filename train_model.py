import os

import joblib
import numpy as np
import pandas as pd
from sklearn.utils import compute_class_weight
from tensorflow.keras.callbacks import ReduceLROnPlateau
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from src.data_preparation import DatasetBuilder
from src.metadata_handler import MetadataHandler
from src.model_builder import ModelBuilder, get_data_generators
import seaborn as sns

os.makedirs("models", exist_ok=True)

IMAGE_SIZE = 224


class TrainModel:
    def __init__(self, model_builder):
        self.model_builder = model_builder

    def generate_classification_report(self, y_true, y_pred, label_encoder):
        """Generate and save comprehensive evaluation reports"""
        class_names = label_encoder.classes_

        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=class_names,
            output_dict=True,
            digits=4
        )

        # Save as CSV
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv('models/classification_report.csv', index=True)

        # Print to console
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))

        # Confusion matrix
        self.plot_confusion_matrix(y_true, y_pred, class_names)

        # Per-class metrics
        self.plot_class_metrics(report_df, class_names)

    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Generate and save confusion matrix visualization"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d',
                    xticklabels=class_names,
                    yticklabels=class_names,
                    cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_class_metrics(self, report_df, class_names):
        """Generate and save per-class metrics visualization"""
        metrics_df = report_df.loc[class_names]

        plt.figure(figsize=(12, 6))
        metrics_df[['precision', 'recall', 'f1-score']].plot(kind='bar')
        plt.title('Per-Class Metrics')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig('models/class_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

    def evaluate_model(self, model, X_test, y_test, label_encoder):
        """Evaluate model and generate reports"""
        # Convert one-hot to class indices
        y_true = np.argmax(y_test, axis=1)
        y_pred = np.argmax(model.predict(X_test), axis=1)

        # Generate reports
        self.generate_classification_report(y_true, y_pred, label_encoder)

        # Calculate and print overall accuracy
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Accuracy: {test_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}")

    def main(self):
        # Initialize metadata handler and dataset builder
        metadata_handler = MetadataHandler("data/image_metadata.csv")
        print(f"Total images in metadata: {len(metadata_handler.metadata)}")
        print(f"Unique species in metadata: {metadata_handler.metadata['species'].nunique()}")

        builder = DatasetBuilder("data/raw_images", metadata_handler)

        # Load and prepare data
        # try:
        X, y_species = builder.load_image_and_labels()

        print(f"NaN in species labels: {np.isnan(y_species).any()}")

        print(f"First 5 species labels: {y_species[:5]}")

        print(f"Loaded {len(X)} images with shapes:")
        print(f"Species labels: {y_species.shape}")

        if len(X) == 0:
            raise ValueError("No images were loaded!")
        if len(y_species) == 0:
            raise ValueError("No species labels were loaded!")
        if len(X) != len(y_species):
            raise ValueError(f"Mismatch between images ({len(X)}) and labels ({len(y_species)})")
        #

        num_classes = len(np.unique(y_species))
        y_species_onehot = to_categorical(y_species, num_classes=num_classes)

        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(X, y_species_onehot,
                                                            test_size=0.2,
                                                            random_state=42,
                                                            stratify=y_species
                                                            )

        from collections import Counter
        print("Label distribution in y_train:", Counter(np.argmax(y_train, axis=1)))
        print("Label distribution in y_test:", Counter(np.argmax(y_test, axis=1)))

        # Verify shapes after split
        print(f"\nAfter split:")
        print(f"X_train shape: {X_train.shape}")

        train_gen, val_gen = get_data_generators('data/dataset')
        model = self.model_builder.build_snake_model()
        # model = self.model_builder.build_dnn(X_train)

        early_stop = [
            EarlyStopping(
                monitor='val_accuracy',  # You can monitor 'val_accuracy' instead
                patience=8,  # Number of epochs to wait after no improvement
                restore_best_weights=True,  # Restore weights from the best epoch
                verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
        ]
        y_train_labels = np.argmax(y_train, axis=1)
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
        class_weight_dict = dict(enumerate(class_weights))

        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30,
                  batch_size=8, callbacks=[early_stop])
        #  , class_weight=class_weight_dict)

        print("\nModel Evaluation:")

        label_encode = builder.label_encoder

        self.evaluate_model(model, X_test, y_test, label_encode)

        joblib.dump(model, "models/snake_identifier_model.pkl")
        joblib.dump(builder.label_encoder, "models/species_encoder.pkl")


if __name__ == "__main__":
    metadata_handler = MetadataHandler("data/image_metadata.csv")
    model_builder = ModelBuilder(metadata_handler)
    train_mode = TrainModel(model_builder)
    train_mode.main()
