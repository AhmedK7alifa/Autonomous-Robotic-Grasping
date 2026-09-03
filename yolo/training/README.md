# YOLO dataset preparation and training records

`capture_dataset.py` and `annotate_dataset.py` are the camera capture and
single-box annotation utilities. New captures are written to the ignored
`data/yolo_capture` staging directory.

`cube_dataset.yaml` points to the dataset in `data/yolo`.
`split_manifest.json` records the 90/15/15 split, `args.yaml` contains the
saved Ultralytics run configuration, and `results.csv` contains epoch
metrics. `figures` includes the selected training history, normalized confusion
matrix, precision-recall curve, and F1 curve.
