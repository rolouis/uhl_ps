import json
import logging
import multiprocessing
import os
import subprocess
import tempfile

import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import differential_evolution

max_workers = multiprocessing.cpu_count() * 2
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
JP2_COMPRESS_PATH = "openjpeg/build/bin/opj_compress"
JP2_DECOMPRESS_PATH = "openjpeg/build/bin/opj_decompress"
test_filename = "example.png"
target_jpeg_qualities = [i for i in range(1, 100, 5)]
target_jpeg_qualities = [80]
miter = 100
# encoder enum
class Encoder:
    JP2 = "jp2"
    JXL = "jxl"
    JPG = "jpg"
    HEIF = "heif"
    WEBP = "webp"
    AVIF = "avif"
    BPG = 'bpg'


def compress_img(img_path, encoder: Encoder, quality=None, level=None, quanitizer=None, keep_file=True) -> int:
    """
    Compresses an image to jp2 format, returns the size of the compressed file in bytes
    :param encoder:
    :param keep_file:
    :param img_path:
    :param quality:
    :return:
    """
    logger.info(f"Compressing {img_path} with {encoder} quality {quality}")
    suffix = ""
    if encoder == Encoder.JP2:
        suffix = ".jp2"
    elif encoder == Encoder.WEBP:
        suffix = ".webp"
    elif encoder == Encoder.JXL:
        suffix = ".jxl"
    elif encoder == Encoder.JPG:
        suffix = ".jpg"
    elif encoder == Encoder.HEIF:
        suffix = ".heif"
    elif encoder == Encoder.AVIF:
        suffix = ".avif"
    elif encoder == Encoder.BPG:
        suffix = ".bpg"
        assert quanitizer is not None
        assert level is not None
        assert quality is None
    else:
        raise ValueError(f"Unknown encoder {encoder}")

    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        if encoder == Encoder.JP2:
            cmd = [JP2_COMPRESS_PATH, "-i", img_path, "-o", tmp.name, "-q", str(quality)]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        elif encoder == Encoder.WEBP:
            cmd = ["cwebp", "-q", str(quality), img_path, "-o", tmp.name]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        elif encoder == Encoder.JXL:
            cmd = ["cjxl", img_path, tmp.name, "-q", str(quality)]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        elif encoder == Encoder.JPG:
            cmd = ["convert", img_path, "-quality", str(quality), tmp.name]

            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        elif encoder == Encoder.HEIF:
            cmd = ["heif-enc", "-q", str(quality), "-o", tmp.name, img_path]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        elif encoder == Encoder.AVIF:
            cmd = ["heif-enc", "-q", str(quality), "-o", tmp.name, "--avif", img_path]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        elif encoder == Encoder.BPG:
            cmd = ["bpgenc", "-m", str(level), "-q", str(quanitizer), "-o", tmp.name, img_path]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            raise ValueError(f"Unknown encoder {encoder}")

        print(" ".join(cmd))
        return os.path.getsize(tmp.name)


EXPERIMENT_ENCODERS = [Encoder.JP2, Encoder.JXL, Encoder.HEIF, Encoder.WEBP, Encoder.AVIF, Encoder.JPG]

EXPERIMENT_ENCODERS = [Encoder.BPG, Encoder.JP2,  Encoder.HEIF]

# EXPERIMENT_ENCODERS = [Encoder.HEIF]

def get_nearest_quality(file_size: int, encoder: Encoder, img_path=test_filename):
    """
    Returns the quality that is closest to the given file size
    :param file_size:
    :param encoder:
    :return:
    """

    # use scipy to find the nearest quality with an optimization algorithm

    def objective(quality):
        compressed_size = compress_img(img_path, encoder, quality[0])
        logger.info(f"Trying quality {quality[0]} Error: {abs(compressed_size - file_size)}")
        if compressed_size > file_size:
            res = abs(compressed_size - file_size) * 10
        else:
            res = abs(compressed_size - file_size)
        # logger.info(f"Trying quality {quality[0]} Error: {res}")
        # Update the plot during the optimization

        return res

    def bpg_objective(params):
        quantizer = params[0]
        level = params[1]
        compressed_size = compress_img(img_path, encoder, level=level, quanitizer=quantizer)
        logger.info(f"Trying quality {level} {quantizer} Error: {abs(compressed_size - file_size)}")
        if compressed_size > file_size:
            res = abs(compressed_size - file_size) * 10
        else:
            res = abs(compressed_size - file_size)
        return res

    if encoder == Encoder.BPG:
        quanizier_bounds = [(0,49)]
        level_bounds = [(1, 9)]
        res = differential_evolution(
            bpg_objective,
            bounds=quanizier_bounds + level_bounds,
            strategy='best1bin', maxiter=miter, disp=False
        )
        return ",".join(str(x) for x in [res.x[0], res.x[1]])
    else:
        bounds = [(0, 100)]
        res = differential_evolution(objective, bounds,
                                     strategy='best1bin', maxiter=miter, disp=False)
        return res.x[0]


def packed_compress_func(args):
    return compress_img(*args)


def nearest_quality_worker(v):
    row, file_path = v
    return get_nearest_quality(row["file_size"], row["encoder"], file_path)


def get_target_qualities(file_path):
    # Get target JPEG sizes in parallel

    with multiprocessing.Pool(processes=max_workers) as pool:
        target_jpeg_sizes = pool.map(packed_compress_func,
                                     [(file_path, Encoder.JPG, quality) for quality in target_jpeg_qualities])
    # Create the dataframe
    quality_df = pd.DataFrame(
        {
            "encoder": [encoder for encoder in EXPERIMENT_ENCODERS for quality in target_jpeg_qualities],
            "quality": [quality for quality in target_jpeg_qualities for encoder in EXPERIMENT_ENCODERS],
            "file_size": [file_size for file_size in target_jpeg_sizes for encoder in EXPERIMENT_ENCODERS],
        }
    )

    # Add the nearest quality for each encoder

    with multiprocessing.Pool(processes=max_workers) as pool:
        quality_df["nearest_quality"] = pool.map(nearest_quality_worker,
                                                 [(row, file_path) for _, row in quality_df.iterrows()])

    # quality_df["nearest_quality"] = quality_df.apply(
    #     lambda row: get_nearest_quality(row["file_size"], row["encoder"], file_path),
    #     axis=1)
    # also add the  error in file size
    quality_df["error"] = quality_df.apply(
        lambda row: compress_img(file_path, row["encoder"], quality=row["nearest_quality"]) - row["file_size"] if row['encoder'] != Encoder.BPG else
        compress_img(file_path, row["encoder"], quanitizer=int(float(row["nearest_quality"].split(",")[1])), level=int(float(row["nearest_quality"].split(",")[0]))) - row["file_size"],
        axis=1)

    quality_df["result_file_size"] = quality_df.apply(
        lambda row: compress_img(file_path, row["encoder"], row["nearest_quality"]) if row['encoder'] != Encoder.BPG else
        compress_img(file_path, row["encoder"], level=int(float(row["nearest_quality"].split(",")[0])), quanitizer=int(float(row["nearest_quality"].split(",")[1]))),
        axis=1)

    return quality_df.to_json(orient="records")


def batch_calculate_target_qualities(png_paths: list):
    results = []
    for path in png_paths:
        res_dict = get_target_qualities(path)
        res_dict = {path: json.loads(res_dict)}
        results.append(res_dict)
    return results


if __name__ == '__main__':
    ## draw a plot filesize per quality per encoder
    for encoder in [ Encoder.HEIF, Encoder.JP2]:
        plt.plot([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                 [compress_img(test_filename, encoder, quality) for quality in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]],
                 label=encoder)
    plt.legend()
    plt.show()

    for l in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        plt.plot(range(0, 50),
                        [compress_img(test_filename, Encoder.BPG, level=l, quanitizer=qant_val) for qant_val in range(0, 50)],
                        label=f"level: {l}")
    plt.legend()
    plt.show()


    # d = batch_calculate_target_qualities(["example.png"])
    # # df from json
    # df = pd.DataFrame.from_dict(d[0]["example.png"])
    # print(df)
    # plot the results
    # for enc in EXPERIMENT_ENCODERS:
    #     df = pd.DataFrame.from_dict(d[0]["example.png"])
    #     df = df[df["encoder"] == enc]
    #     plt.plot(df["quality"], df["result_file_size"], label=enc)
    #