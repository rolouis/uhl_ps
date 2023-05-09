import pydantic
import sewar
import cv2


def get_metrics(img1_path, img2_path):
    """
    SSIM, SCC, PSNR, ERGAS,
    mse
    rmse
    psnr
    rmse_sw
    uqi
    ssim
    ergas
    scc
    rase
    sam
    msssim
    vifp
    psnrb
    :param img1_path:
    :param img2_path:
    :return:
    """
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    return {
        "mse": sewar.full_ref.mse(img1, img2),
        "rmse": sewar.full_ref.rmse(img1, img2),
        "psnr": sewar.full_ref.psnr(img1, img2),
        "uqi": sewar.full_ref.uqi(img1, img2),
        "ssim": sewar.full_ref.ssim(img1, img2),
        "ergas": sewar.full_ref.ergas(img1, img2),
        "scc": sewar.full_ref.scc(img1, img2),
        "rase": sewar.full_ref.rase(img1, img2),
        "sam": sewar.full_ref.sam(img1, img2),
        "msssim": sewar.full_ref.msssim(img1, img2),
        "vifp": sewar.full_ref.vifp(img1, img2),
    }


if __name__ == "__main__":
    example_img_path = "../images/example.png"
    example_img = cv2.imread(example_img_path)
    # print(sewar.full_ref.ssim(example_img, example_img))
    print(get_metrics(example_img_path, example_img_path))
    pass
