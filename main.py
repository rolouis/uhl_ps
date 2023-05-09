from matplotlib import pyplot as plt

from mediasecurity.target_attributes import wsefgnerginnipgwe, get_target_qualities
from utils.Encoders import Encoder
from utils.approximation import compress_img, get_nearest_quality, decode_img
from utils.imgtools import get_image_size
from utils.metrics import get_metrics

def get_target_values(files):
    """

    :param files: Array of image paths
    :return: Json of target values
    """
    res_dict = {}
    for f in files:
        res_dict[f] = get_target_qualities(f)
    return res_dict

if __name__ == "__main__":
    import glob
    print(get_target_values(glob.glob("images/*.png")))
