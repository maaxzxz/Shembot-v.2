# Optional custom-model training

The website does not require this heavy training environment. These scripts are
included to make the completed data and model work reproducible.

Use Python 3.9:

```powershell
py -3.9 -m venv .venv-train
.\.venv-train\Scripts\Activate.ps1
python -m pip install -r training\requirements-train.txt
python training\prepare_voc.py --input path\to\exported_voc --output data\cctv_indoor_clean
python training\train_efficientdet_lite.py --dataset data\cctv_indoor_clean --epochs 30 --batch-size 4
```

The source preparation script groups augmented images before splitting and
repairs invalid box coordinates. The training script loads Pascal VOC pairs,
trains EfficientDet-Lite0, evaluates the locked test split, and exports a
metadata-bearing TFLite model.

