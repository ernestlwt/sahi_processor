# SAHI_as_processing
Using SAHI as a pre and post processing step

## Impetus
To make it easier to use sahi without changes to the main model inference code.

## How to use

```
processor = SAHIProcessing()
slice_info, batched_images = processor.get_slice_batches(list_of_images)

# slice_info will explain how to stitch the prediction output back
# run batched_images through your model and output predictions
# combine all batch of predictions into  List[List[l, t, r, b, score, class_id]]

results = processor.run_sahi_algo(slice_info, predictions)

```

## Format for slice info
```
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

# list_position must be ordered
# original_shape and resize_shape will only exist when to_slice is False

```

## Format for predictions
```
# List of images predictions
[
    [ 
        [l, t, r, b, conf, class_id],
        [l, t, r, b, conf, class_id], ...
    ],...
]
```
