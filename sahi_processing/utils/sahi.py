"""
This file consist of function from the official sahi repo
"""

from typing import Optional, List, Tuple

def get_slice_bboxes(
    image_height: int,
    image_width: int,
    slice_height: Optional[int] = None,
    slice_width: Optional[int] = None,
    auto_slice_resolution: bool = True,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
) -> List[List[int]]:
    """
    https://github.com/obss/sahi/blob/main/sahi/slicing.py

    Slices `image_pil` in crops.
    Corner values of each slice will be generated using the `slice_height`,
    `slice_width`, `overlap_height_ratio` and `overlap_width_ratio` arguments.

    Args:
        image_height (int): Height of the original image.
        image_width (int): Width of the original image.
        slice_height (int, optional): Height of each slice. Default None.
        slice_width (int, optional): Width of each slice. Default None.
        overlap_height_ratio(float): Fractional overlap in height of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        overlap_width_ratio(float): Fractional overlap in width of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        auto_slice_resolution (bool): if not set slice parameters such as slice_height and slice_width,
            it enables automatically calculate these params from image resolution and orientation.

    Returns:
        List[List[int]]: List of 4 corner coordinates for each N slices.
            [
                [slice_0_left, slice_0_top, slice_0_right, slice_0_bottom],
                ...
                [slice_N_left, slice_N_top, slice_N_right, slice_N_bottom]
            ]
    """
    slice_bboxes = []
    y_max = y_min = 0

    if slice_height and slice_width:
        y_overlap = int(overlap_height_ratio * slice_height)
        x_overlap = int(overlap_width_ratio * slice_width)
    elif auto_slice_resolution:
        x_overlap, y_overlap, slice_width, slice_height = get_auto_slice_params(height=image_height, width=image_width)
    else:
        raise ValueError("Compute type is not auto and slice width and height are not provided.")

    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - slice_width)
                ymin = max(0, ymax - slice_height)
                slice_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes

def get_auto_slice_params(height: int, width: int) -> Tuple[int, int, int, int]:
    """
    https://github.com/obss/sahi/blob/main/sahi/slicing.py

    According to Image HxW calculate overlap sliding window and buffer params
    factor is the power value of 2 closest to the image resolution.
        factor <= 18: low resolution image such as 300x300, 640x640
        18 < factor <= 21: medium resolution image such as 1024x1024, 1336x960
        21 < factor <= 24: high resolution image such as 2048x2048, 2048x4096, 4096x4096
        factor > 24: ultra-high resolution image such as 6380x6380, 4096x8192
    Args:
        height:
        width:

    Returns:
        slicing overlap params x_overlap, y_overlap, slice_width, slice_height
    """
    resolution = height * width
    factor = calc_resolution_factor(resolution)
    if factor <= 18:
        return get_resolution_selector("low", height=height, width=width)
    elif 18 <= factor < 21:
        return get_resolution_selector("medium", height=height, width=width)
    elif 21 <= factor < 24:
        return get_resolution_selector("high", height=height, width=width)
    else:
        return get_resolution_selector("ultra-high", height=height, width=width)





