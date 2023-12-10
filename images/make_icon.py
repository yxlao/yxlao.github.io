from pathlib import Path
import camtools as ct
import argparse
import numpy as np
from typing import Tuple


def process_image(im_src: np.ndarray, dimension: int, pad_color="white") -> np.ndarray:
    """
    Crop white boarders, pad image to square, resize to dimension.

    Args:
        im_src: (H, W, 3), float32.
        dimension: int, dimension of the square output image.
        pad_color: str, color of the padding, must be "white" or "black".

    Returns:
        im_dst: (dimension, dimension, 3), float32.
    """
    # Check input image.
    if im_src.dtype != np.float32:
        raise TypeError("Input image must be float32.")
    if im_src.ndim != 3 or im_src.shape[2] != 3:
        raise ValueError("Input image must be (H, W, 3).")
    if pad_color not in ["white", "black"]:
        raise ValueError('pad_color must be "white" or "black".')

    # Crop white boarders.
    crop_u, crop_d, crop_l, crop_r = ct.image.compute_cropping(im_src)
    im_dst = ct.image.apply_cropping_padding(
        im_src,
        cropping=(crop_u, crop_d, crop_l, crop_r),
        padding=(0, 0, 0, 0),
    )

    # Pad to square.
    # ct.image.pad_to_square(im_dst) is NOT available, implemnt it here.
    h, w = im_dst.shape[:2]
    pad_val = 1.0 if pad_color == "white" else 0.0
    if h > w:
        pad_l = (h - w) // 2
        pad_r = h - w - pad_l
        im_dst = np.pad(
            im_dst,
            ((0, 0), (pad_l, pad_r), (0, 0)),
            mode="constant",
            constant_values=pad_val,
        )
    elif h < w:
        pad_u = (w - h) // 2
        pad_d = w - h - pad_u
        im_dst = np.pad(
            im_dst,
            ((pad_u, pad_d), (0, 0), (0, 0)),
            mode="constant",
            constant_values=pad_val,
        )

    # Resize.
    im_dst = ct.image.resize(im_dst, shape_wh=(dimension, dimension))
    im_dst = im_dst.astype(np.float32)

    return im_dst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Input image files.")
    parser.add_argument(
        "--dimension",
        "-d",
        help="Dimension of the output images.",
        type=int,
        default=320,
    )
    args = parser.parse_args()

    for input_path in args.inputs:
        im_src_path = Path(input_path)
        im_dst_path = im_src_path.parent / f"{im_src_path.stem}.jpg"
        if not im_src_path.is_file():
            print(f"File not found: {im_src_path}, skipping.")
            continue

        im_src = ct.io.imread(im_src_path, alpha_mode="white")
        im_dst = process_image(im_src, args.dimension)
        ct.io.imwrite(im_dst_path, im_dst)
        print(f"Saved processed image to {im_dst_path}.")


if __name__ == "__main__":
    main()
