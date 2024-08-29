from sahi_processor.sahi_processor import SAHIProcessor
import cv2

my_image = cv2.imread("tests/data/small-vehicles1.jpeg")

processor = SAHIProcessor(sahi_postprocess_type="NMM", sahi_postprocess_match_metric="IOS")
batched_images = processor.get_slice_batches([my_image], model_batchsize=4)

# store slices in folder for sanity check
count = 0
for b in batched_images:
    for img in b:
        cv2.imwrite("tests/data/output/" + str(count) + ".jpg", img)
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

results = processor.run_sahi_algo([my_image], mock_predictions)
print(results)