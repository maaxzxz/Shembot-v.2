# Data card — CCTV Indoor Person Detection v1

## Source and intended use

The dataset establishes a one-class person-detection baseline for SHEMBOT. It
was supplied by a Roboflow Universe user under **CC BY 4.0**:
<https://universe.roboflow.com/yolo-k9iku/cctv-indoor-person-detection-bi2vp>.

## Clean source-grouped split

| Split | Source groups | Images | Person boxes |
| --- | ---: | ---: | ---: |
| Train | 438 | 1,072 | 8,332 |
| Validation | 61 | 145 | 1,051 |
| Locked test | 53 | 133 | 1,056 |
| **Total** | **552** | **1,350** | **10,439** |

The only class is `person`. Images are paired with Pascal VOC XML annotations.
The audit found no missing pairs and removed no boxes. It clamped 203
out-of-bounds annotations to valid image coordinates.

## Leakage prevention

Roboflow created three augmented versions of many originals. The preparation
script derives a pre-augmentation source ID and assigns every version of that
source to one deterministic split. Therefore, a crop or rotation of one scene
cannot leak across train, validation, and test. The test set is not used for
threshold tuning.

## Preprocessing

- Roboflow export: EXIF auto-orientation and 640 × 640 stretch.
- Source augmentation: crop 0–15%, rotation ±15°, brightness ±25%, Gaussian
  blur 0–1.1 pixels, and 1.37% salt-and-pepper noise.
- Custom EfficientDet input: 320 × 320 RGB via TFLite Model Maker.
- Live browser input: selected crop, automatic rotation, aspect-preserving
  downscale to a maximum 512-pixel side, then MediaPipe preprocessing.

## Known gaps

The public set is dominated by elevated, distant CCTV views and does not match
portrait phone video, close partial bodies, motion, backlighting, or the real
deployment background. The next dataset must include consented phone frames and
hard negatives such as empty rooms, furniture, fans, vehicles, pets, posters,
mannequins, clothing, roofs, and shadows.

