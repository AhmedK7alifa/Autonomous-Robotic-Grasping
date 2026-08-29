# YOLO training metadata

## Dataset

The frozen single-class cube dataset contained 120 images with matching labels: 90 train, 15 validation, and 15 held-out test. Raw images and labels are intentionally not distributed in this repository. [split_manifest.json](split_manifest.json) preserves the deterministic allocation and file-level hashes; [cube_dataset.yaml](cube_dataset.yaml) preserves the original lab path and split names.

The capture and semi-automatic annotation utilities are included unchanged. Candidate annotations were visually reviewed in the experiment workflow.

## Training configuration

No standalone training script existed in the frozen project. Training was performed through Ultralytics, and [args.yaml](args.yaml) is the authoritative saved configuration:

- architecture: yolo26n, initialized from a pretrained checkpoint;
- maximum epochs: 150;
- early-stopping patience: 30;
- image size: 640;
- batch size: 4;
- optimizer: AdamW;
- device: CPU;
- workers: 0;
- deterministic seed: 42.

Training stopped after epoch 59, with best validation performance at epoch 29. The reported best-checkpoint validation metrics were precision 1.000, recall 0.932, mAP@0.5 0.991, and mAP@0.5:0.95 0.916. The epoch history is preserved in [results.csv](results.csv).

The frozen checkpoint is [../final/model/best.pt](../final/model/best.pt), SHA-256 32C32FCEF42B92A5ED3FD37F0023281251FBE50D94342AF65AD6614592BFF7C6.

Because the raw dataset is excluded, detector retraining is not self-contained in this public snapshot.
