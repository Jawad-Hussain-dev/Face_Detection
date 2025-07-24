import cv2
import numpy as np
from skimage.feature import hog

def extract_hog_features(image_path, resize_size=(256, 256), visualize=False):
    """
    Extract HOG features from a grayscale image.

    Args:
        image_path (str): Path to image file.
        resize_size (tuple): Size to resize the image to (width, height).
        visualize (bool): Whether to return the HOG image for display.

    Returns:
        np.ndarray: HOG feature vector (and HOG image if visualize=True)
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be read.")
    
    image = cv2.resize(image, resize_size)

    features, hog_image = hog(image,
                              orientations=9,
                              pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2),
                              block_norm='L2-Hys',
                              visualize=True,
                              feature_vector=True)

    return (features, hog_image) if visualize else features


