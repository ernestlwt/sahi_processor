from enum import Enum
from typing import Any, Dict, Optional, List

import cv2
import numpy as np

from utils.misc import split_list_into_batches
from utils.sahi import get_slice_bboxes

POSTPROCESSING_ALGO = [
    "GREEDYNMM",
    "NMM",
    "NMS"
]

POSTPROCESSING_METRIC = [
    "IOS",
    "IOU"
]

class SAHIProcessing():
    """
    """
    model_batchsize: int = 8
    model_input_height: int = 240
    model_input_width: int = 320

    sahi_image_height_threshold: int = 900
    sahi_image_width_threshold: int = 900
    sahi_slice_height: int = 240
    sahi_slice_width: int = 320
    sahi_overlap_height_ratio: float = 0.3
    sahi_overlap_width_ratio: float = 0.3
    sahi_postprocess_match_algo: str = "GREEDYNMM"
    sahi_postprocess_match_metric: str = "IOS"
    sahi_postprocess_match_threshold: float = 0.5
    sahi_postprocess_class_agnostic: bool = True
    sahi_auto_slice_resolution: bool = True,
    sahi_include_full_frame: bool = True
    sahi_crop_full_frame: bool = True

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        assert self.sahi_postprocess_match_algo in POSTPROCESSING_ALGO, "Invalid algo, please choose among: " + str(POSTPROCESSING_ALGO)
        assert self.sahi_postprocess_match_metric in POSTPROCESSING_METRIC, "Invalid metric, please choose among: " + str(POSTPROCESSING_METRIC) 

    def get_slice_info(self, list_of_images: List[np.ndarray]) -> Dict:
        """
        This function takes in a list of images and calculate how much to slice them
        """
        sliced_list = []
        for i, image in enumerate(list_of_images):
            image_h, image_w, _ = image.shape

            if image_h > self.sahi_image_height_threshold or image_w > self.sahi_image_width_threshold:

                if self.sahi_include_full_frame:
                    sliced_list.append({"list_position": i, "to_slice": False})

                slice_bboxes = get_slice_bboxes(
                    image_h,
                    image_w,
                    self.sahi_slice_height,
                    self.sahi_slice_width,
                    overlap_height_ratio=self.sahi_overlap_height_ratio,
                    overlap_width_ratio=self.sahi_overlap_width_ratio,
                    auto_slice_resolution=self.sahi_auto_slice_resolution
                )
                for s_b in slice_bboxes:
                    sliced_list.append({
                        "list_position": i, 
                        "to_slice": True,
                        "ltrb": s_b
                    })
                    print(s_b)
            else:
                sliced_list.append({"list_position": i, "to_slice": False})
        return sliced_list

    def get_slice_batches(self, list_of_images: List[np.ndarray], slice_info: Dict) -> List[List[np.ndarray]]:
        """
        This function takes in a list of images and the slice info generated from  get_slice_info and organize them in batches
        """
        batches_of_info = split_list_into_batches(slice_info, self.model_batchsize)

        batches_of_images = []
        for batch in batches_of_info:
            batch_of_image = []
            for info in batch:
                if info["to_slice"]:
                    ltrb  = info["ltrb"]
                    batch_of_image.append(list_of_images[info["list_position"]][ ltrb[1]:ltrb[3], ltrb[0]:ltrb[2]])
                else:
                    if self.sahi_crop_full_frame:
                        cropped_image = cv2.resize(list_of_images[info["list_position"]], (self.sahi_image_width_threshold, self.sahi_image_height_threshold))
                        batch_of_image.append(cropped_image)
                    else:
                        batch_of_image.append(list_of_images[info["list_position"]])
            batches_of_images.append(batch_of_image)
    
        return batches_of_images





def main():
    processor = SAHIProcessing(sahi_postprocess_match_algo="NMS", sahi_postprocess_match_metric="IOU")

    my_image = cv2.imread("test/data/small-vehicles1.jpeg")

    slice_info = processor.get_slice_info([my_image])
    batched_images = processor.get_slice_batches([my_image], slice_info)

    count = 0
    for b in batched_images:
        for img in b:
            cv2.imwrite("test/data/output/" + str(count) + ".jpg", img)
            count += 1
    

if __name__ == "__main__":
    main()
