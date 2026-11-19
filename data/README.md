# Dataset location and layout

The complete 111 MB source ZIP is kept outside Git to avoid repository bloat:

`CCTV Indoor Person Detection.v1i.voc.zip`

Source: <https://universe.roboflow.com/yolo-k9iku/cctv-indoor-person-detection-bi2vp>  
License: **CC BY 4.0**

After running `training/prepare_voc.py`, the local layout is:

```text
data/cctv_indoor_clean/
|-- train/   # 1,072 JPG/XML pairs; 438 source groups
|-- valid/   #   145 JPG/XML pairs;  61 source groups
|-- test/    #   133 JPG/XML pairs;  53 source groups (locked)
|-- manifest.csv
`-- audit.json
```

Every augmented variant of one original image stays in the same split. The
included `manifest.csv` records source ID, original Roboflow split, clean split,
and box count. The included audit is in `evidence/dataset_audit.json`.

