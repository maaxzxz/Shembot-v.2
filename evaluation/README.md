# Reproducible 30-image validation baseline

This folder contains one image from each of 30 distinct validation source
groups, its Pascal VOC annotation, the quantized CPU model, and the evaluator.
It is deliberately separate from the locked test split.

From the repository root:

```powershell
python -m pip install -r requirements.txt
python -m unittest evaluation/test_evaluate_baseline.py -v
python evaluation/evaluate_baseline.py
```

The command writes:

- `evaluation/results/baseline_summary.json`
- `evaluation/results/baseline_results.csv`
- `evaluation/results/baseline_preview.jpg`

Green boxes in the preview are predictions; orange boxes are ground truth.
Precision, recall, and F1 are calculated with one-to-one matching at IoU 0.50.
The 30 selected images are validation evidence only and are not used for the
custom model's held-out test metrics.

