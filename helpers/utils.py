import cv2
import numpy as np


def read_image(image_path):
    """
    reads image from path and shape it into (c,h,w)
    """
    img = cv2.imread(image_path, 0)  # Assuming the image is grayscale
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add channel dimension
    return img_array
