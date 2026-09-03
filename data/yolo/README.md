# Wooden-cube detection dataset

This is the single-class YOLO dataset used by the experiment:

| Split | Images | Labels |
|---|---:|---:|
| Train | 90 | 90 |
| Validation | 15 | 15 |
| Test | 15 | 15 |
| Total | 120 | 120 |

Every image shows the fixed experimental workspace and wooden cube. Labels use
class `0` (`cube`) and normalized YOLO bounding-box coordinates. Split
membership and hashes are recorded in `yolo/training/split_manifest.json`.
