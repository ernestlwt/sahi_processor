from enum import Enum
from typing import Any, Dict, Optional, List

import cv2
import numpy as np

from utils.misc import split_list_into_batches

from sahi.slicing import get_slice_bboxes
from sahi.prediction import ObjectPrediction
from sahi.postprocess.utils import ObjectPredictionList
from sahi.postprocess.combine import NMSPostprocess, NMMPostprocess, GreedyNMMPostprocess

POSTPROCESSING_ALGO = {
    "GREEDYNMM": GreedyNMMPostprocess,
    "NMM": NMMPostprocess,
    "NMS": NMSPostprocess
}

POSTPROCESSING_METRIC = [
    "IOS",
    "IOU"
]

class SAHIProcessing():
    """
    """
    model_batchsize: int = 8

    sahi_image_height_threshold: int = 900
    sahi_image_width_threshold: int = 900
    sahi_slice_height: int = 400
    sahi_slice_width: int = 400
    sahi_overlap_height_ratio: float = 0.3
    sahi_overlap_width_ratio: float = 0.3
    sahi_postprocess_match_algo: str = "GREEDYNMM"
    sahi_postprocess_match_metric: str = "IOS"
    sahi_postprocess_match_threshold: float = 0.5
    sahi_postprocess_class_agnostic: bool = True
    sahi_auto_slice_resolution: bool = True,
    sahi_include_full_frame: bool = True
    sahi_resize_full_frame: bool = True

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        assert self.sahi_postprocess_match_algo in POSTPROCESSING_ALGO.keys(), "Invalid algo, please choose among: " + str(POSTPROCESSING_ALGO.keys())
        assert self.sahi_postprocess_match_metric in POSTPROCESSING_METRIC, "Invalid metric, please choose among: " + str(POSTPROCESSING_METRIC) 

        self.sahi_postprocessing_function = POSTPROCESSING_ALGO[self.sahi_postprocess_match_algo](
            match_threshold=self.sahi_postprocess_match_threshold,
            match_metric=self.sahi_postprocess_match_metric,
            class_agnostic=self.sahi_postprocess_class_agnostic
        )

    def get_slice_info(self, list_of_images: List[np.ndarray]) -> Dict:
        """
        This function takes in a list of images and calculate how much to slice them
        """
        sliced_list = []
        for i, image in enumerate(list_of_images):
            image_h, image_w, _ = image.shape

            if image_h > self.sahi_image_height_threshold or image_w > self.sahi_image_width_threshold:

                if self.sahi_include_full_frame:
                    sliced_list.append({
                        "list_position": i, 
                        "to_slice": False, 
                        "original_shape":[image_w, image_h], 
                        "resized_shape": [self.sahi_slice_width, self.sahi_slice_height] if self.sahi_resize_full_frame else [image_w, image_h]})

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
                    if self.sahi_resize_full_frame:
                        cropped_image = cv2.resize(list_of_images[info["list_position"]], (self.sahi_slice_width, self.sahi_slice_height))
                        batch_of_image.append(cropped_image)
                    else:
                        batch_of_image.append(list_of_images[info["list_position"]])
            batches_of_images.append(batch_of_image)
    
        return batches_of_images

    def merge_slice_predictions(self, slice_info: Dict, list_of_predictions: List[List[float]]) -> List[List[float]]:
        assert len(slice_info) == len(list_of_predictions), "Length of slice_info and list_of_predictions does not match"
        merged_predictions = []

        for info, preds in zip(slice_info, list_of_predictions):
            list_position = info["list_position"]

            if info["to_slice"]:
                for pred in preds:
                    pred[0] = info["ltrb"][0] + pred[0]
                    pred[1] = info["ltrb"][1] + pred[1]
                    pred[2] = info["ltrb"][0] + pred[2]
                    pred[3] = info["ltrb"][1] + pred[3]
            elif info["original_shape"][0] != info["resized_shape"][0] or info["original_shape"][1] != info["resized_shape"][1]:
                multiplier_w = info["original_shape"][0] / info["resized_shape"][0]
                multiplier_h = info["original_shape"][1] / info["resized_shape"][1]

                for pred in preds:
                    # write a function for this
                    pred[0] = multiplier_w * pred[0]
                    pred[1] = multiplier_h * pred[1]
                    pred[2] = multiplier_w * pred[2]
                    pred[3] = multiplier_h * pred[3]

            if len(merged_predictions) -1 < list_position:
                merged_predictions.append(preds)
            else:
                merged_predictions[list_position].extend(preds)

        return merged_predictions

    def convert_to_sahi_predictions(self, predictions):
        batch_list = []
        for batch in predictions:
            pred_list= []
            for img_pred in batch:
                pred_list.append(ObjectPrediction(bbox=img_pred[0:4], score=img_pred[4], category_id=img_pred[5]))
            batch_list.append(pred_list)
        return batch_list

    def run_sahi_algo(self, list_of_predictions: List[ObjectPrediction]):
        result = self.sahi_postprocessing_function(list_of_predictions)
        return result


def main():
    processor = SAHIProcessing(sahi_postprocess_match_algo="NMM", sahi_postprocess_match_metric="IOS")

    my_image = cv2.imread("test/data/small-vehicles1.jpeg")

    slice_info = processor.get_slice_info([my_image])
    batched_images = processor.get_slice_batches([my_image], slice_info)

    count = 0
    for b in batched_images:
        for img in b:
            cv2.imwrite("test/data/output/" + str(count) + ".jpg", img)
            count += 1
    
    # simulated predictions for small-vehicles1.jpeg at 400:400 slices (left most blue car)
    mock_predictions = [
        [
            [120,221,145,251,0.56,0]
        ],
        [
            [320,319,385,364,0.8,0]
        ],
        [
            [37,319,108,360,0.73,0]
        ],
        [],
        [],
        [
            [321,139,386,181,0.72,0]
        ],
        [
            [37,140,106,178,0.9,0]
        ],
        [],
        []
    ]

    merged_p = processor.merge_slice_predictions(slice_info, mock_predictions)
    list_of_objectpredictions = processor.convert_to_sahi_predictions(merged_p)
    results = processor.run_sahi_algo(list_of_objectpredictions[0])
    print(results)


if __name__ == "__main__":
    main()
