import multiprocessing
import os

import pandas as pd

from utils.Encoders import Encoder
from utils.approximation import (
    packed_compress_func,
    nearest_quality_worker,
    compress_img,
)


def get_target_qualities(
    file_path,
    target_jpeg_qualities=[80, 85, 90, 95, 100],
    experiment_encoders=[
        Encoder.JP2,
        Encoder.JXR,
        Encoder.JXL,
        Encoder.JPG,
        Encoder.HEIF,
        Encoder.WEBP,
        Encoder.AVIF,
        Encoder.BPG,
    ],
    max_workers=8,
):
    # Get target JPEG sizes in parallel

    target_jpeg_sizes = get_jpeg_targetsizes(
        file_path, max_workers, target_jpeg_qualities
    )
    # Create the dataframe
    quality_df = pd.DataFrame(
        {
            "encoder": [
                encoder
                for encoder in experiment_encoders
                for quality in target_jpeg_qualities
            ],
            "quality": [
                quality
                for quality in target_jpeg_qualities
                for encoder in experiment_encoders
            ],
            "file_size": [
                file_size
                for file_size in target_jpeg_sizes
                for encoder in experiment_encoders
            ],
        }
    )

    # Add the nearest quality for each encoder

    with multiprocessing.Pool(processes=max_workers) as pool:
        quality_df["nearest_quality"] = pool.map(
            nearest_quality_worker,
            [(row, file_path) for _, row in quality_df.iterrows()],
        )

    quality_df["error"] = quality_df.apply(
        lambda row: compress_img(
            file_path, row["encoder"], quality=row["nearest_quality"]
        )
        - row["file_size"]
        if row["encoder"] != Encoder.BPG
        else compress_img(
            file_path,
            row["encoder"],
            quanitizer=int(float(row["nearest_quality"].split(",")[1])),
            level=int(float(row["nearest_quality"].split(",")[0])),
        )
        - row["file_size"],
        axis=1,
    )

    quality_df["result_file_size"] = quality_df.apply(
        lambda row: compress_img(file_path, row["encoder"], row["nearest_quality"])
        if row["encoder"] != Encoder.BPG
        else compress_img(
            file_path,
            row["encoder"],
            level=int(float(row["nearest_quality"].split(",")[0])),
            quanitizer=int(float(row["nearest_quality"].split(",")[1])),
        ),
        axis=1,
    )

    return quality_df.to_json(orient="records")


def get_jpeg_targetsizes(file_path, max_workers, target_jpeg_qualities):
    """
    Get target JPEG sizes in parallel for given qualities and image
    :param file_path:
    :param max_workers:
    :param target_jpeg_qualities:
    :return:
    """
    with multiprocessing.Pool(processes=max_workers) as pool:
        target_jpeg_sizes = pool.map(
            packed_compress_func,
            [(file_path, Encoder.JPG, quality) for quality in target_jpeg_qualities],
        )
    return dict(zip(target_jpeg_qualities, target_jpeg_sizes))


def get_folder_input_stats(dir_path, target_sizes=[80, 85, 90, 95, 100]):
    """
    Get the target sizes for all images in a folder (JPEG Quality)
    :param dir_path:
    :param target_sizes:
    :return:
    """
    files = os.listdir(dir_path)
    files = [os.path.join(dir_path, file) for file in files if file.endswith(".png")]
    rdict = {}
    for f in files:
        rdict[f] = get_jpeg_targetsizes(f, 8, target_sizes)
    return rdict


def wsefgnerginnipgwe():
    """

    :return:
    """
    import pandas as pd

    input_stats = get_folder_input_stats("images")
    for key, val in input_stats.items():
        get_target_qualities(
            key,
            experiment_encoders=[
                Encoder.JP2,
                Encoder.JXR,
                Encoder.JXL,
                Encoder.JPG,
                Encoder.HEIF,
                Encoder.WEBP,
                Encoder.AVIF,
                Encoder.BPG,
            ],
        )
        input_stats[key]["target_sizes"] = val
    print(input_stats)
