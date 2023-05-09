from matplotlib import pyplot as plt

from mediasecurity.target_attributes import wsefgnerginnipgwe
from utils.Encoders import Encoder
from utils.approximation import compress_img, get_nearest_quality, decode_img
from utils.imgtools import get_image_size
from utils.metrics import get_metrics

if __name__ == "__main__":
    wsefgnerginnipgwe()
    exit()
    ## draw a plot filesize per quality per encoder
    # i = [i * 0.1 for i in range(1, 1001)]
    # for encoder in [Encoder.JXR]:
    #     plt.plot(i,
    #              [compress_img(test_filename, encoder, quality * 0.01) for quality in i],
    #              label=encoder)
    # plt.legend()
    # # plt.show()
    test_file = "images/example.png"
    # fsize = get_image_size(test_file)
    # print(fsize)
    # # target: 1/2 of original filesize
    # target = fsize / 20
    # qual = get_nearest_quality(target, Encoder.JP2, test_file)
    # print(qual)
    # print(f"Target {target} result {compress_img(test_file, Encoder.JPG,qual )} ")
    for i in range(1, 100):
        img_path = compress_img(test_file, Encoder.JXR, i, keep=True)[1]
        print(
            f"Quality {i} result { get_metrics(test_file, decode_img(img_path, Encoder.JXR))} "
        )
