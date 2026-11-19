# Milestone 2 — 50% checkpoint evidence map

## Definition-of-done checklist

- [x] Fresh setup instructions in `README.md`
- [x] Deployed HTTPS prototype opens directly from the README link
- [x] CPU/browser inference; no GPU required
- [x] Exact runtime/model artifacts are stored locally and hashed
- [x] Source-grouped train/validation/test dataset audit
- [x] Locked test split documented
- [x] Data card with source, CC BY 4.0 license, counts, labels, and limitations
- [x] Preprocessing code path and before/after visual evidence
- [x] Reproducible baseline with evaluator, model, 30 validation images/annotations, and saved JSON/CSV
- [x] Working live prototype and GitHub Pages workflow
- [x] Automated static smoke test
- [x] Weighted progress report, blockers, and next actions
- [x] Editable Word version of the progress report
- [ ] Replace identity placeholders in `reports/PROGRESS_REPORT.md`
- [x] Exact checkpoint commit published and tagged `cp2-50`

## Teacher-selected run

Clone the tagged checkpoint and reproduce its checks:

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

Then open the deployed prototype at
<https://maaxzxz.github.io/Shembot-v.2/>. The detector, pose model, JavaScript
bundle, and WASM runtime are versioned in `lib/mediapipe`.

## What is intentionally not included

The 111 MB source dataset and generated training caches/checkpoints are not
duplicated in this Git-ready ZIP. `data/manifest.csv`, the audit, source/license
record, a 30-image validation subset, the CPU baseline model/evaluator, saved
metrics, and training scripts are included. The full source ZIP is retained
separately and can be supplied if the teacher requests it.

