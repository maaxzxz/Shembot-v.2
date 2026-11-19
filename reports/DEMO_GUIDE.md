# In-class checkpoint demo

## Before uploading

- [ ] Fill the team identity and owner fields in `PROGRESS_REPORT.md`.
- [ ] Run `node tests/static_smoke.test.mjs`.
- [ ] Open the deployed HTTPS prototype and test camera permission.
- [ ] Keep one well-lit scene and one difficult scene ready.
- [ ] Create and push the Git tag `cp2-50` on this exact checkpoint commit.

## Two-minute sequence

1. Open <https://maaxzxz.github.io/Shembot-v.2/> in a current browser.
2. Start the camera and show person-only boxes, live count, FPS, and capacity.
3. Show camera switching and the detection-area control.
4. Explain the full/zoom scan, pose verification, duplicate merging, and
   cross-frame confirmation in `src/js/app.js`.
5. Open the data card and preprocessing comparison.
6. State the two measured baselines and why the custom CCTV model is not yet
   the live default.

## Likely questions

**What is the input size?**  
The browser downsizes each selected crop so its longest side is at most 512
pixels. The custom EfficientDet training run uses 320 × 320 RGB.

**How is leakage prevented?**  
All augmentations of one original source image share an ID and remain in one
train/validation/test split.

**Was test data used for tuning?**  
No. The 133-image custom-model test split is locked. Future thresholds must be
tuned on validation data only.

**How are non-person objects rejected?**  
The detector has a person-only allowlist; ordinary candidates are checked with
pose, geometry, temporal persistence, and score rules before being counted.

**How are duplicate body parts handled?**  
The app applies NMS, containment and proximity comparisons, pose association,
and cross-scan consolidation before updating tracks and the displayed count.

**Why can mobile loading take time?**  
The first visit must load and compile approximately 56 MB of local model/WASM
assets. Later visits normally benefit from browser caching; device CPU, memory,
storage, and connection speed still affect startup.

**What remains after 50%?**  
Phone-specific data collection, retraining, failure regression tests, threshold
tuning, device benchmarking, and final locked evaluation.

