# Preprocessing and feature path

The live preprocessing implementation is in `src/js/app.js`, primarily
`getCrop()`, `getScanAreas()`, and `prepareInferenceFrame()`.

For every scan the app:

1. selects the full frame or user-defined detection area;
2. adds four overlapping zoom tiles to improve small-person recall;
3. tests the selected orientation candidates;
4. downscales without upscaling so the longest side is at most 512 pixels;
5. draws into an inference canvas for MediaPipe's detector;
6. maps boxes back through scale, crop, rotation, and mirroring;
7. filters to `person`, verifies pose when available, and merges duplicates.

Training-only Roboflow augmentation is documented in `DATA_CARD.md`; it is not
applied to live prediction. The visual comparison at
`evidence/preprocessing_comparison.jpg` uses a licensed validation frame and
shows the original image, full-frame 512-pixel preparation, and one 58% zoom
scan. Its source image is retained as `evidence/preprocessing_source.jpg`.

The detector model itself performs its required tensor conversion and
normalization inside the vendored MediaPipe Tasks runtime. Keeping that runtime
and models in this ZIP makes the inference preprocessing version reproducible.

