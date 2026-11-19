# Model evaluation and deployment decision

## Reproducible baselines

| Experiment | Evaluation set | Result |
| --- | --- | --- |
| **Included OpenCV YOLOX-S INT8 baseline** | **30 images from 30 distinct validation source groups** | **Precision 0.7273; recall 0.0773; F1 0.1397 at IoU 0.50; count MAE 6.1667; 7.248 FPS** |
| OpenCV HOG/SVM count baseline | 300 validation frames | MAE 20.92; RMSE 21.5177; exact match 0%; mean count accuracy 34.1177%; 17.6476 FPS |
| Custom EfficientDet-Lite0, epoch 23 | 133 held-out test images | AP 0.0958; AP50 0.2886; AP75 0.0345; AR@100 0.2830 |

The first row can be reproduced entirely from this repository with
`python evaluation/evaluate_baseline.py`. Its per-image CSV, preview, and raw
JSON are committed under `evaluation/results/`. The other machine-readable
experiment records are included under `evidence/`.

## Interpretation

Both baselines are useful but insufficient for final deployment. The count
baseline undercounts crowded scenes. The custom model performs better at IoU
0.5 than across stricter thresholds but still has low small-person recall. Its
training domain is elevated CCTV, while the product uses handheld or mounted
phone cameras.

## Checkpoint decision

The custom epoch-23 model is preserved as a training artifact, but this ZIP
keeps the generic EfficientDet-Lite0 detector as the live default. The live app
adds pose verification, person-only filtering, full/zoom scans, geometry
checks, cross-frame confirmation, and duplicate consolidation. The final model
decision will be made only after a phone-domain locked test set exists.

This report does not treat FPS or training accuracy as proof of detection
quality. Accuracy claims are tied to the named validation/test data and saved
metrics.

