# SHEMBOT v2 — Milestone 2 (50% checkpoint)

SHEMBOT is a mobile-first, browser-based people counter. It opens a phone or
laptop camera, runs person detection locally, draws boxes, stabilizes the
count, and displays capacity and entry/exit information. Camera frames are not
uploaded by the application.

This repository is the actual 50% checkpoint implementation. The live website,
vendored MediaPipe runtime, detector model, pose model, checkpoint reports,
dataset audit, training scripts, measured baselines, and smoke tests are all
included.

## Fresh-clone checkpoint run

> **✓ A fresh clone plus the README steps runs the prototype.**

Clone the tagged checkpoint and verify its included files and validation
baseline:

```powershell
git clone --branch cp2-50 https://github.com/maaxzxz/Shembot-v.2.git
cd Shembot-v.2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
node tests/static_smoke.test.mjs
python -m unittest evaluation/test_evaluate_baseline.py -v
python evaluation/evaluate_baseline.py
```

Open the deployed HTTPS prototype—no local server is required:

**<https://maaxzxz.github.io/Shembot-v.2/>**

Allow camera access and press **Start camera**. Use a current Chrome, Edge, or
Safari browser. No npm install or build step is required; `npm test` is an
optional shortcut for the JavaScript smoke test.

## Implemented at this checkpoint

- MediaPipe EfficientDet-Lite0 person proposals, filtered to class `person`
- lightweight PoseLandmarker verification
- portrait/landscape orientation testing
- full-frame plus overlapping zoom scans for small people
- NMS, containment-aware duplicate suppression, and cross-frame tracking
- configurable room capacity and detection area
- front/rear camera switching and mirrored preview correction
- lighting and clarity guidance
- entry/exit line counter and local CSV history
- local correct/incorrect feedback capture
- static GitHub Pages deployment workflow

## Reproducible evidence

| Evidence | Result |
| --- | --- |
| Count baseline | 300 validation frames: MAE 20.92, RMSE 21.5177, 17.6476 FPS |
| Custom EfficientDet-Lite0 | 133-image held-out test: AP 0.0958, AP50 0.2886, AR@100 0.2830 |
| Detection dataset | 1,350 images, 10,439 person boxes, source-grouped split |
| Included validation baseline | 30 distinct validation sources: precision 0.7273, recall 0.0773, F1 0.1397 at IoU 0.50 |
| Browser smoke test | Confirms all app assets, models, required UI elements, and person-only filters |

The custom CCTV model is documented as a baseline but is not the deployed
default. Its public CCTV domain does not match the intended phone-camera domain
well enough. The checkpoint therefore uses the vendored generic detector plus
pose, geometry, temporal, and duplicate checks while phone-specific training
data is collected.

## Checkpoint documents

- `CHECKPOINT_50.md` — submission checklist and evidence map
- `reports/DATA_CARD.md` — source, license, split, labels, preprocessing, gaps
- `reports/MODEL_EVALUATION.md` — measured results and deployment decision
- `reports/PREPROCESSING.md` — code path and visual evidence
- `reports/PROGRESS_REPORT.md` — weighted 50% progress and blockers
- `reports/PROGRESS_REPORT.docx` — formatted, editable Word submission copy
- `reports/DEMO_GUIDE.md` — in-class demo steps and likely questions

## Project structure

```text
.
|-- index.html
|-- assets/css/styles.css
|-- src/js/
|-- lib/mediapipe/
|-- training/
|-- data/
|-- evidence/
|-- evaluation/
|-- reports/
|-- tests/static_smoke.test.mjs
|-- package.json
`-- requirements.txt
```

## Privacy

Detection, pose verification, tracking, and history remain in the browser.
Incorrect samples are downloaded only when the user explicitly requests it;
they are never uploaded automatically.

