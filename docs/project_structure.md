# Project structure

The repository separates byte-identical experimental artifacts from newly authored publication documentation.

| Path | Role | Provenance |
|---|---|---|
| README.md | Main academic overview | Newly authored from frozen evidence |
| LICENSE | GNU Affero General Public License, version 3 | Standard license text |
| THIRD_PARTY_NOTICES.md | Upstream attribution and licensing notice | Newly authored |
| CITATION.cff | Citation metadata | Newly authored from approved metadata |
| requirements.txt | Documented direct Python dependencies | Reconstructed from environment records |
| environment.yml | Compact Windows-oriented environment | Reconstructed from environment records |
| aruco | ArUco method documentation | Actuation controllers excluded at the public safety boundary |
| yolo/final | Publication-safe YOLO reference subset, including `best.pt` | Included artifacts are byte-identical; actuation code and pycache excluded |
| yolo/training | Dataset metadata, saved args/history, capture and annotation utilities | Byte-identical copies plus README |
| yolo/evaluation | Preserved validation utilities and checkpoint note | Byte-identical copies plus README |
| calibration/camera | Camera calibration code and frozen outputs | Byte-identical copies |
| mapping/final | Day-7 model, source observations, builder, validator | Byte-identical copies; one observation file renamed |
| results/aruco | 20 trial JSON records, summaries, analyzer | Byte-identical copies |
| results/yolo | 20 trial JSON records, summaries, promoter, analyzer | Byte-identical copies |
| results/comparison | Official comparison outputs and script | Byte-identical copies |
| results/figures | Mapping visualization | Byte-identical copy |
| docs | Setup, protocol, reproducibility, and structure | Newly authored |
| paper | Preprint status and citation guidance | Newly authored |

## Publication-layout tradeoff

The authoritative source used numbered development directories and several frozen reference copies. The public repository uses method-oriented directories to make the research record understandable. Every included experimental source artifact remains byte-identical. The provenance map also records seven controller artifacts that were deliberately withheld at the public safety boundary.

## Deliberate exclusions

- raw YOLO images and labels;
- ChArUco capture images;
- per-trial ArUco camera images;
- development failures, temporary diagnostics, and duplicate backups;
- caches, pycache, compiled Python files, and environment directories;
- videos and unrelated logs;
- per-unit robot calibration that should not be applied to another robot; and
- executable robot-actuation controllers and instructions for direct physical robot motion; and
- any material outside the frozen two-method cube comparison.

PUBLIC_REPOSITORY_REVIEW.md provides the full source-to-public mapping and final validation status.
