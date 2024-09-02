# SAHI_processor
Using SAHI as a pre and post processing step

Link to original repo, [SAHI: Slicing Aided Hyper Inference](https://github.com/obss/sahi)

## Impetus
To make it easier to use sahi without changes to the your model inference code. Also allow for batched inference which is something that is not provided in the official repo.

## How to use
Install package with 
```
pip install sahi_processor
```

Sample usage
```
from sahi_processor.sahi_processor import SAHIProcessor

processor = SAHIProcessor()
batched_images = processor.get_slice_batches(list_of_images, model_batchsize=batchsize)

# run batched_images through your model and output predictions
# combine all batches of predictions into  List[List[l, t, r, b, score, class_id]]

merged_predictions = processor.run_sahi_algo(list_of_images, predictions)
```

A sample test script can be ran via `python tests/test.py`

## Parameters and defaults for SAHIProcesor

```
# image_height_threshold: int
#     only do sahi if the height of the image exceeds this
# image_width_threshold: int
#     only do sahi if the width of the image exceeds this
#     The rationale for these 2 parameters is that since this repo is meant to batch multiple images,
#     there could be a case where we have images of different size in a batch. For image that may be
#     around the size of a single slice, there may not be a point to slice them
# resize_full_frame: bool
#     to resize images that has not been sliced size to the slice size if True else no resizing

image_height_threshold: int = 600 
image_width_threshold: int = 600
resize_full_frame: bool = True

# the variables below are from the default sahi repo: https://github.com/obss/sahi/blob/main/sahi/predict.py#L125
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
```

## Formats to note
`list_of_images` is a list of cv2 images in `(H, W, C)`

`predictions` is a list of predictions for each image.
Below is a sample:
```
[
    [ 
        [l, t, r, b, score, class_id],
        [l, t, r, b, score, class_id], ...
    ],...
]
```
