from matplotlib import pyplot as plt

from mediasecurity.target_attributes import wsefgnerginnipgwe, get_target_qualities
from utils.Encoders import Encoder
from utils.approximation import compress_img, get_nearest_quality, decode_img
from utils.imgtools import get_image_size
from utils.metrics import get_metrics

if __name__ == "__main__":
    test_file = "images/example.png"
    df = get_target_qualities(
        test_file,
        target_jpeg_qualities=[40, 50],
        experiment_encoders=[Encoder.JP2, Encoder.HEIF],
    )
    print(df)
