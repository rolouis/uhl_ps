import numpy as np
from PIL import Image
import cv2
def get_image_sizes(image_path):
    # load image using cv2.imread()
    img = cv2.imread(image_path)

    # get dimensions of the image
    height, width, channels = img.shape

    # get number of bits per channel
    bits_per_channel = 8

    # return dimensions and bits per channel as a tuple
    return (height, width, channels, bits_per_channel)


def get_image_size(img_path):
    """
    Get the image size in bytes, calculated by W * H * C * Channeldepth
    :param img_path: path to image
    :return: Filesize in bytes
    """

    return np.prod(get_image_sizes(img_path)) / 8




