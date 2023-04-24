import json
import logging
import multiprocessing
import os
import subprocess
import tempfile

import pandas as pd
from scipy.optimize import differential_evolution

from utils.Encoders import Encoder

max_workers = multiprocessing.cpu_count() * 2
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

test_filename = "images/example.png"
miter = 1000

JP2_COMPRESS_PATH = "openjpeg/build/bin/opj_compress"
JP2_DECOMPRESS_PATH = "openjpeg/build/bin/opj_decompress"


# encoder enum

def decode_img(img_path, encoder: Encoder) -> str:
    """
    Decode an image using the given encoder
    :param img_path:
    :param encoder:
    :return:
    """
    # get the file suffix
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        if encoder == Encoder.JP2:
            cmd = [JP2_DECOMPRESS_PATH, "-i", img_path, "-o", tmp.name]
        elif encoder == Encoder.WEBP:
            cmd = ["dwebp", "-i", img_path, "-o", tmp.name]
        elif encoder == Encoder.JXL:
            cmd = ["djxl", img_path, tmp.name]


        elif encoder == Encoder.JXR:
            # jxrencapp takes as input .bmp or .tif images
            tmp_file2 = tempfile.NamedTemporaryFile(suffix='.bmp', delete=False)
            # tmp_file2.close()
            cmd = ["jxrdecapp", "-i", img_path, "-o", tmp_file2.name]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            cmd = ["convert", tmp_file2.name, tmp.name]



        elif encoder == Encoder.JPG:
            cmd = ["convert", img_path, tmp.name]

        elif encoder == Encoder.HEIF:
            cmd = ["heif-convert", tmp.name, img_path]

        elif encoder == Encoder.AVIF:
            cmd = ["heif-convert", tmp.name, img_path]

        elif encoder == Encoder.BPG:
            cmd = ["bpgdec", "-o", tmp.name, img_path]

        else:
            raise ValueError(f"Unknown encoder {encoder}")

        print(" ".join(cmd))
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return str(tmp.name)


def compress_img(img_path, encoder: Encoder, quality=None, level=None, quanitizer=None, keep=False) -> int:
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
    encoder_suffix = {
        Encoder.JP2: ".jp2",
        Encoder.WEBP: ".webp",
        Encoder.JXL: ".jxl",
        Encoder.JXR: ".jxr",
        Encoder.JPG: ".jpg",
        Encoder.HEIF: ".heif",
        Encoder.AVIF: ".avif",
        Encoder.BPG: ".bpg"
    }
    suffix = encoder_suffix.get(encoder, "")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=not keep) as tmp:
        if encoder == Encoder.JP2:
            cmd = [JP2_COMPRESS_PATH, "-i", img_path, "-o", tmp.name, "-q", str(quality)]
        elif encoder == Encoder.WEBP:
            cmd = ["cwebp", "-q", str(quality), img_path, "-o", tmp.name]
        elif encoder == Encoder.JXL:
            cmd = ["cjxl", img_path, tmp.name, "-q", str(quality)]

        elif encoder == Encoder.JXR:
            # jxrencapp takes as input .bmp or .tif images
            tmp_file2 = tempfile.NamedTemporaryFile(suffix='.bmp')
            # tmp_file2.close()

            cmd_toBmp = ["convert", img_path, tmp_file2.name]
            subprocess.run(cmd_toBmp)
            qual = float(quality)
            # -q range [1,255] where 1 is lossless (quantization)
            cmd = ["jxrencapp", "-i", tmp_file2.name, "-o", tmp.name, "-q", str(qual)]

        elif encoder == Encoder.JPG:
            cmd = ["convert", img_path, "-quality", str(quality), tmp.name]
        elif encoder == Encoder.HEIF:
            cmd = ["heif-enc", "-q", str(quality), "-o", tmp.name, img_path]
        elif encoder == Encoder.AVIF:
            cmd = ["heif-enc", "-q", str(quality), "-o", tmp.name, "--avif", img_path]
        elif encoder == Encoder.BPG:
            cmd = ["bpgenc", "-m", str(level), "-q", str(quanitizer), "-o", tmp.name, img_path]

        else:
            raise ValueError(f"Unknown encoder {encoder}")
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(" ".join(cmd))
        print("Out name: ", tmp.name)
        if keep:
            return os.path.getsize(tmp.name), str(tmp.name)
        return os.path.getsize(tmp.name)


# EXPERIMENT_ENCODERS = [Encoder.JP2, Encoder.JXL, Encoder.JXR, Encoder.HEIF, Encoder.WEBP, Encoder.AVIF, Encoder.JPG]

# EXPERIMENT_ENCODERS = [Encoder.BPG, Encoder.JP2, Encoder.JXR, Encoder.HEIF]

EXPERIMENT_ENCODERS = [Encoder.JXR]


def get_nearest_quality(file_size: int, encoder: Encoder, img_path=test_filename):
    """
    Returns the quality that is closest to the given file size
    :param img_path:
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
        quantizer_bounds = [(0, 49)]
        level_bounds = [(1, 9)]
        res = differential_evolution(
            bpg_objective,
            bounds=quantizer_bounds + level_bounds,
            strategy='best1bin', maxiter=miter, disp=False
        )
        return ",".join(str(x) for x in [res.x[0], res.x[1]])

    elif encoder == Encoder.JXR:
        quantizer_bounds = [(0.0, 1.0)]
        res = differential_evolution(objective, quantizer_bounds,
                                     strategy='best1bin', maxiter=miter, disp=False)
        return res.x[0]

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




def batch_calculate_target_qualities(png_paths: list):
    results = []
    for path in png_paths:
        res_dict = get_target_qualities(path)
        res_dict = {path: json.loads(res_dict)}
        results.append(res_dict)
    return results
