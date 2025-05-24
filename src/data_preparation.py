import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

IMAGE_SIZE = (224, 224)


class DatasetBuilder:
    def __init__(self, data_dir, metadata_handler):
        self.data_dir = data_dir  # data/raw_images
        self.metadata_handler = metadata_handler
        self.label_encoder = LabelEncoder()

    def load_image_and_labels(self):
        images = []
        species_labels = []

        for species, img_files in tqdm(self.metadata_handler.get_all_species().items()):
            for img_file in img_files:
                try:
                    # Load image
                    img = cv2.imread(img_file)
                    img = cv2.resize(img, IMAGE_SIZE)
                    images.append(img)
                    species_labels.append(species)
                except Exception as e:
                    print(f"Skipping {img_file} due to error: {str(e)}")
                    continue

        images = np.array(images) / 255.0
        species_labels = self.label_encoder.fit_transform(species_labels)
        return images, species_labels













    # def get_snake_augmenter(self):
    #     """Specialized augmentation for snake images"""
    #     return ImageDataGenerator(
    #         rotation_range=25,
    #         width_shift_range=0.1,
    #         height_shift_range=0.1,
    #         shear_range=0.1,
    #         zoom_range=0.15,
    #         horizontal_flip=True,
    #         vertical_flip=False,
    #         fill_mode='reflect',
    #         brightness_range=[0.9, 1.1],
    #         channel_shift_range=10.0
    #     )
    #
    # def hood_expansion_transform(self, img):
    #     """Simulates cobra hood flaring"""
    #     img = np.array(img)
    #     h, w = img.shape[:2]
    #
    #     # Create expansion mask (elliptical)
    #     y, x = np.ogrid[:h, :w]
    #     center_y, center_x = h // 2, w // 2
    #     mask = ((x - center_x) ** 2 / (0.3 * w) ** 2 + (y - center_y) ** 2 / (0.4 * h) ** 2) <= 1
    #
    #     # Expand hood region
    #     expanded = cv2.resize(img, (int(w * 1.3), int(h * 1.3)))
    #     expanded = expanded[int(h * 0.15):int(h * 1.15), int(w * 0.15):int(w * 1.15)]
    #
    #     # Blend with original
    #     result = np.where(mask[..., None], expanded, img)
    #     return Image.fromarray(np.uint8(result))
    #
    # def sharpen_details(self, img, factor=1.5):
    #     """Enhances scale patterns and head markings"""
    #     img = np.array(img)
    #     kernel = np.array([[-1, -1, -1],
    #                        [-1, 9 * factor, -1],
    #                        [-1, -1, -1]])
    #     sharpened = cv2.filter2D(img, -1, kernel)
    #     return Image.fromarray(np.clip(sharpened, 0, 255).astype('uint8'))
    #
    # def _augment_king_cobra(self, img):
    #     """Special augmentations for King Cobras"""
    #     # Simulate hood expansion
    #     if random.random() > 0.5:
    #         img = self.hood_expansion_transform(img)
    #     # Enhance head patterns
    #     img = self.sharpen_details(img, factor=1.5)
    #     return img
    #
    # def _augment_cobra(self, img):
    #     """Special augmentations for other cobras"""
    #     img = self.sharpen_details(img, factor=1.2)
    #     return img
