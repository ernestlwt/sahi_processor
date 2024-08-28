# SAHI_as_processing
Using SAHI as a pre and post processing step

## Impetus
To make it easier to use sahi without changes to the main model inference code.


## Format for slice definition
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
[
    [l, t, r, b, conf, class_id]
]
```
