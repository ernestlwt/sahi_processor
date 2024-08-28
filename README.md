# SAHI_processor
Using SAHI as a pre and post processing step

## Impetus
To make it easier to use sahi without changes to the main model inference code.

## How to use

```
processor = SAHIProcessing()
batched_images = processor.get_slice_batches(list_of_images)

# run batched_images through your model and output predictions
# combine all batch of predictions into  List[List[l, t, r, b, score, class_id]]

results = processor.run_sahi_algo(list_of_images, predictions)
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
