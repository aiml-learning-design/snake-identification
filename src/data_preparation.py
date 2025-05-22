import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from tqdm import tqdm
import cv2
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = (224, 224)


class DatasetBuilder:
    def __init__(self, data_dir, metadata_handler):
        self.data_dir = data_dir
        self.metadata_handler = metadata_handler
        self.label_encoder = LabelEncoder()
        self.venom_encoder = MultiLabelBinarizer()
        self.location_encoder = MultiLabelBinarizer()
        self.toxicity_encoder = MultiLabelBinarizer()
        self.species_specific_augment = {
            "Ophiophagus hannah": self._augment_king_cobra,
            "Naja naja": self._augment_cobra
        }
        self.augmenter = self.get_snake_augmenter()

    def _augment_king_cobra(self, img):
        """Special augmentations for King Cobras"""
        # Simulate hood expansion
        if random.random() > 0.5:
            img = self.hood_expansion_transform(img)
        # Enhance head patterns
        img = self.sharpen_details(img, factor=1.5)
        return img

    def _augment_cobra(self, img):
        """Special augmentations for other cobras"""
        img = self.sharpen_details(img, factor=1.2)
        return img

    def load_image_and_labels(self):
        images = []
        species_labels = []
        venom_labels = []
        geo_labels = []
        toxicity_labels = []

        for species, img_files in tqdm(self.metadata_handler.get_all_species().items()):
            for img_file in img_files:
                try:
                    # Load image
                    with Image.open(img_file) as img:
                        # Convert to RGB if needed
                        if img.mode != 'RGB':
                            img = img.convert('RGB')

                        # Convert to numpy array and check dimensions
                        img_array = np.array(img)

                        if len(img_array.shape) == 2:
                            img_array = np.stack((img_array,)*3, axis=-1)

                        # Handle problematic images (1x1x3 float32 or other unusual cases)
                        if img_array.shape == (1, 1, 3):
                            if img_array.dtype == np.float32:
                                # Convert float32 [0,1] to uint8 [0,255]
                                color = (img_array[0, 0] * 255).astype(np.uint8)
                            else:
                                color = img_array[0, 0]
                            # Create solid color image
                            img_array = np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8)
                            img_array[:, :] = color
                            img = Image.fromarray(img_array)
                        elif img_array.ndim != 3 or img_array.shape[2] != 3:
                            # Create blank image for other invalid cases
                            img_array = np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8)
                            img = Image.fromarray(img_array)
                        else:
                            # Normal case - ensure proper type
                            if img_array.dtype != np.uint8:
                                img_array = (img_array * 255).astype(
                                    np.uint8) if img_array.max() <= 1.0 else img_array.astype(np.uint8)
                            img = Image.fromarray(img_array)

                        # Apply species-specific augmentation first
                        if species in self.species_specific_augment:
                            img = self.species_specific_augment[species](img)

                        # Convert to array for augmentation
                        img_array = np.array(img)

                        # Apply general augmentations
                        augmented_array = self.augmenter.random_transform(img_array.astype(np.float32) / 255.0)
                        augmented_array = (augmented_array * 255).astype(np.uint8)
                        img = Image.fromarray(augmented_array)

                        # Resize and final checks
                        img = img.resize(IMAGE_SIZE)
                        img_array = np.array(img)

                        if img_array.shape != (*IMAGE_SIZE, 3) or img_array.dtype != np.uint8:
                            # Create blank image if still problematic
                            img_array = np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8)

                        images.append(img_array)
                        species_labels.append(species)
                        venom_labels.append(self.metadata_handler.get_venom_type(species).split(", "))
                        geo_labels.append(self.metadata_handler.get_species_info(species).get("region", "").split(", "))
                        toxicity_labels.append(self.metadata_handler.get_toxicity_level(species))

                except Exception as e:
                    print(f"Skipping {img_file} due to error: {str(e)}")
                    continue

            # Verify we have data
        if len(images) == 0:
            raise ValueError("No images were successfully loaded!")
        if len(images) != len(species_labels):
            raise ValueError(f"Mismatch: {len(images)} images but {len(species_labels)} species labels")

        # Encode labels
        species_labels_encoded = self.label_encoder.fit_transform(species_labels)
        venom_labels_encoded = self.venom_encoder.fit_transform(venom_labels)
        geo_labels_encoded = self.location_encoder.fit_transform(geo_labels)
        toxicity_label_encoded = self.toxicity_encoder.fit_transform(toxicity_labels)

        return (
            np.array(images),
            species_labels_encoded,
            venom_labels_encoded,
            geo_labels_encoded,
            toxicity_label_encoded
        )

    def get_snake_augmenter(self):
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

    def hood_expansion_transform(self, img):
        """Simulates cobra hood flaring"""
        img = np.array(img)
        h, w = img.shape[:2]

        # Create expansion mask (elliptical)
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        mask = ((x - center_x) ** 2 / (0.3 * w) ** 2 + (y - center_y) ** 2 / (0.4 * h) ** 2) <= 1

        # Expand hood region
        expanded = cv2.resize(img, (int(w * 1.3), int(h * 1.3)))
        expanded = expanded[int(h * 0.15):int(h * 1.15), int(w * 0.15):int(w * 1.15)]

        # Blend with original
        result = np.where(mask[..., None], expanded, img)
        return Image.fromarray(np.uint8(result))

    def sharpen_details(self, img, factor=1.5):
        """Enhances scale patterns and head markings"""
        img = np.array(img)
        kernel = np.array([[-1, -1, -1],
                           [-1, 9 * factor, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(img, -1, kernel)
        return Image.fromarray(np.clip(sharpened, 0, 255).astype('uint8'))
