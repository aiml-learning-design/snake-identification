import cv2
import numpy as np


def preprocess_image(image_array: np.ndarray, target_size=(224, 224)) -> np.ndarray:
    """Process numpy array for model input"""
    try:
        # Resize and validate channels
        image = cv2.resize(image_array, target_size)

        # Handle channel conversions
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 1:  # Single channel
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Normalize and validate dtype
        image = image.astype(np.float32) / 255.0

        # Ensure 3 channels (H, W, 3)
        assert image.shape == (*target_size, 3), f"Invalid shape: {image.shape}"
        return image

    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")


def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges


def display_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
