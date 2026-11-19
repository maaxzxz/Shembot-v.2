# CCTV person detector training report

## Dataset preparation

- Source: `CCTV Indoor Person Detection.v1i.voc.zip`
- Original pairs: 1,350 images and Pascal VOC annotations
- Clean grouped split: 1,072 train, 145 validation, 133 test
- Person boxes: 8,332 train, 1,051 validation, 1,056 test
- Corrected out-of-bounds boxes: 203
- Augmented variants were grouped by source image to prevent split leakage.

## Model

- Architecture: EfficientDet-Lite0
- Input: 320 x 320 RGB
- Export: quantized TFLite with object-detector metadata
- Maximum detections: 50
- Best frozen-backbone checkpoint: epoch 23

## Held-out test comparison

| Run | AP | AP50 | AP small | AP medium | AP large | AR max 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Epoch 30 | 0.0951 | 0.2897 | 0.0135 | 0.0890 | 0.1704 | 0.2740 |
| Epoch 23 | **0.0958** | 0.2886 | 0.0146 | **0.0890** | 0.1759 | **0.2830** |
| Whole-model fine-tune | 0.0810 | 0.2895 | **0.0373** | 0.0666 | **0.1829** | 0.2225 |

## Deployment decision

Epoch 23 is the best overall artifact, but AP and small-person recall are not
high enough to replace the website's current detector safely. The public CCTV
data is dominated by elevated, distant, grayscale scenes and does not match the
phone camera's color, angle, lighting, occlusion, and close-range false
positives. The model is preserved for testing, but is not deployed by default.

The next useful dataset should contain labeled frames from the actual phone
camera positions, plus hard-negative frames containing chairs, fans, roofs,
vehicles, pets, clothing, and empty rooms. Those examples are necessary to
reduce both missed people and non-person detections.
