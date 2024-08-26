# SAHI_as_processing
Using SAHI as a pre and post processing step

## Format for slice definition
```
[
    {
        "list_position": 0,
        "to_slice": False,
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
```