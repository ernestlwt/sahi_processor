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

class SAHIProcessor():
    """
    Class that uses the SAHI algorithm to slice an entire image into smaller slices,
    batching the slices into the model batchsize to make inference more efficient, 
    and then merging the sliced predictions back with reference to the entire image

    The class variables mostly come from get_sliced_prediction() at https://github.com/obss/sahi/blob/main/sahi/predict.py
    Do refer to the official code for their explanation

    Below are some explanation for the variables that have been added for this repo

    Parameters:
    image_height_threshold: int
        only do sahi if the height of the image exceeds this
    image_width_threshold: int
        only do sahi if the width of the image exceeds this
        The rationale for these 2 parameters is that since this repo is meant to batch multiple images,
        there could be a case where we have images of different size in a batch. For image that may be
        around the size of a single slice, there may not be a point to slice them
    resize_full_frame: bool
        to resize images that has not been sliced size to the slice size if True else no resizing
    """
    image_height_threshold: int = 600
    image_width_threshold: int = 600
    resize_full_frame: bool = True

    sahi_slice_height: int = 400
    sahi_slice_width: int = 400
    sahi_overlap_height_ratio: float = 0.3
    sahi_overlap_width_ratio: float = 0.3
    sahi_perform_standard_pred: bool = True
    sahi_postprocess_type: str = "GREEDYNMM"
    sahi_postprocess_match_metric: str = "IOS"
    sahi_postprocess_match_threshold: float = 0.5
    sahi_postprocess_class_agnostic: bool = True
    sahi_auto_slice_resolution: bool = True

    def __init__(self, **kwargs):
        """
        Initialize SAHIProcessor
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        assert self.sahi_postprocess_type in POSTPROCESSING_ALGO.keys(), "Invalid algo, please choose among: " + str(POSTPROCESSING_ALGO.keys())
        assert self.sahi_postprocess_match_metric in POSTPROCESSING_METRIC, "Invalid metric, please choose among: " + str(POSTPROCESSING_METRIC) 

        self.sahi_postprocessing_function = POSTPROCESSING_ALGO[self.sahi_postprocess_type](
            match_threshold=self.sahi_postprocess_match_threshold,
            match_metric=self.sahi_postprocess_match_metric,
            class_agnostic=self.sahi_postprocess_class_agnostic
        )

    def _get_slice_info(self, list_of_images: List[np.ndarray]) -> Dict:
        """
        This function takes in a list of images and calculate how much to slice them

        Parameters:
        list_of_images: List[np.ndarray]
            list of images in numpy array format in (H, W, C)

        Returns:
        sliced_info: Dict
            A dictionary that provides information to stich the image predictions back.
            Required for self.run_sahi_algo to merge the predictions

            Sample Format:
            [
                {
                    "list_position": 0,
                    "to_slice": False,
                    "original_shape": [x1, y1],
                    "resized_shape": [x2, y2]
                }, {
                    "list_position": 1,
                    "to_slice": True,
                    "ltrb": [l, t, r, b],
                }, {
                "list_position": 1,
                    "to_slice": True,
                    "ltrb": [l, t, r, b],  
                }, ...
            ]
            list_position must be ordered
            original_shape and resize_shape will only exist when to_slice is False
            ltrb explains is the slice location with respect to the original image
        """
        sliced_info = []
        for i, image in enumerate(list_of_images):
            image_h, image_w, _ = image.shape

            if image_h > self.image_height_threshold or image_w > self.image_width_threshold:

                if self.sahi_perform_standard_pred:
                    sliced_info.append({
                        "list_position": i, 
                        "to_slice": False, 
                        "original_shape":[image_w, image_h], 
                        "resized_shape": [self.sahi_slice_width, self.sahi_slice_height] if self.resize_full_frame else [image_w, image_h]})

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
                    sliced_info.append({
                        "list_position": i, 
                        "to_slice": True,
                        "ltrb": s_b
                    })
            else:
                sliced_info.append({"list_position": i, "to_slice": False})
        return sliced_info

    def get_slice_batches(self, list_of_images: List[np.ndarray], model_batchsize:int = 1) -> List[List[np.ndarray]]:
        """
        This function takes in a list of images to generate slice info
        and batches the sliced images into the model_batchsize

        Parameters:
        list_of_images: List[np.ndarray]
            list of images in numpy array format in (H, W, C)

        Returns:
        slice_info: Dict
            A dictionary that provides information to stich the image predictions back.
            Required for self.run_sahi_algo to merge the predictions
        
        batched_images: List[List[np.ndarray]]
            List of batches of images in numpy array format in (H, W, C)
        """
        slice_info = self._get_slice_info(list_of_images)

        batches_of_info = split_list_into_batches(slice_info, model_batchsize)

        batched_images = []
        for batch in batches_of_info:
            batch_of_image = []
            for info in batch:
                if info["to_slice"]:
                    ltrb  = info["ltrb"]
                    batch_of_image.append(list_of_images[info["list_position"]][ ltrb[1]:ltrb[3], ltrb[0]:ltrb[2]])
                else:
                    if self.resize_full_frame:
                        cropped_image = cv2.resize(list_of_images[info["list_position"]], (self.sahi_slice_width, self.sahi_slice_height))
                        batch_of_image.append(cropped_image)
                    else:
                        batch_of_image.append(list_of_images[info["list_position"]])
            batched_images.append(batch_of_image)
    
        return batched_images

    def _merge_slice_predictions(self, list_of_images: List[np.ndarray], list_of_predictions: List[List[float]]) -> List[List[float]]:
        """
        Merge prediction output using information from the slice_info dictionary

        Parameters:
        list_of_images: List[np.ndarray]
            list of images in numpy array format in (H, W, C)

        list_of_predictions: List[List[float]]
            A list of sliced images' predictions in [l, t, r, b, score, class_id] format

        Returns:
        merged_predictions: List[List[float]]
            A list of full images' predictions in [l, t, r, b, score, class_id] format
        """
        slice_info = self._get_slice_info(list_of_images)
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

    def run_sahi_algo(self, list_of_images: List[np.ndarray], list_of_predictions: List[List[float]]):
        """
        Merge prediction output using information from the slice_info dictionary

        Parameters:
        list_of_images: List[np.ndarray]
            list of images in numpy array format in (H, W, C)

        list_of_predictions: List[List[float]]
            A list of sliced images' predictions in [l, t, r, b, score, class_id] format

        Returns:
        processed_predictions: List[List[float]]
            A list of full images' prediction that has been processed with the SAHI algorithm
            that has been set when initializing the SAHIProcessor class
        """
        merged_predictions = self._merge_slice_predictions(list_of_images, list_of_predictions)

        processed_predictions = []
        for batch in merged_predictions:
            sahi_pred_list= []
            # convert to SAHI's ObjectPrediction object and run SAHI's post processing
            for img_pred in batch:
                sahi_pred_list.append(ObjectPrediction(bbox=img_pred[0:4], score=img_pred[4], category_id=img_pred[5]))
            processed_sahi_pred_list = self.sahi_postprocessing_function(sahi_pred_list)

            # convert back to prediction format List[List[l, t, r, b, score, conf]]
            pred_list = []
            for s_p in processed_sahi_pred_list:
                pred = processed_sahi_pred_list[0].bbox.to_xyxy()
                pred.extend([processed_sahi_pred_list[0].score.value, processed_sahi_pred_list[0].category.id])
                pred_list.append(pred)

            processed_predictions.append(pred_list)
        return processed_predictions
