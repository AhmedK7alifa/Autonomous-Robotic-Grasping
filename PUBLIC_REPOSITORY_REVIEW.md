# Public repository final publication review

## Status

**PUBLICATION_READY**

Review date: 2026-08-29

This repository has completed the human-approved public-release pass. The approved author identity, GNU AGPL v3 license, third-party attribution, strict two-method scope, frozen research claims, and public robot-safety boundary are now reflected consistently across the release.

No commit, push, publication, remote change, or repository-setting change was performed.

## Approved identity and citation metadata

- Sole author display: **AHMED KHALIFA IDRISS ELHAJ**
- CFF given names: **Ahmed Khalifa Idriss**
- CFF family name: **Elhaj**
- Affiliation: **College of Computer Science, Chengdu Normal University, Chengdu, Sichuan, China**
- Preprint status: **Preprint in preparation.**
- No Chinese name, ORCID, DOI, arXiv identifier, email address, venue, or acceptance status is asserted.

[CITATION.cff](CITATION.cff) parses successfully and contains only the approved identity and release metadata.

## License and third-party attribution

The repository is released under the standard **GNU Affero General Public License, version 3 only**, SPDX identifier **AGPL-3.0-only**. [LICENSE](LICENSE) is the official GNU AGPL v3 text and has:

- size: 34,523 bytes;
- SHA-256: 0D96A4FF68AD6D4B6F1F30F713B18D5184912BA8DD389F86AA7710DB079ABCB0.

[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) attributes YOLO26 and the Ultralytics toolchain to Ultralytics and its contributors, records the upstream AGPL-3.0/enterprise licensing context, and makes no ownership claim over the underlying architecture or toolchain.

The retained [yolo/final/model/best.pt](yolo/final/model/best.pt) file is the project-specific trained YOLO26 checkpoint:

- size: 5,364,357 bytes;
- SHA-256: 32C32FCEF42B92A5ED3FD37F0023281251FBE50D94342AF65AD6614592BFF7C6.

Third-party components remain subject to their upstream licenses and notices.

## Public safety boundary

This public release contains no executable robot-actuation controller and no instructions for direct physical robot motion.

Seven historical controller artifacts remain in the read-only authoritative archive but are absent from the public repository:

- five ArUco V3 location-specific benchmark controllers;
- the YOLO full pick-and-place candidate; and
- the duplicated ArUco controller formerly stored in the YOLO mapping snapshot.

Their source and former public paths remain auditable in [SOURCE_FILE_MAP.tsv](SOURCE_FILE_MAP.tsv) with publication status removed_public_safety_boundary. The included dry-run/reference programs contain perception, mapping, validation, or non-actuating plan logic only; the final executable-pattern scan found no LeRobot hardware import, follower construction, robot connection, or action-send call.

Hardware-specific poses and calibration files are retained only as research records. The documentation does not present them as transferable commands or operating instructions.

## Scope

The public scientific scope is exactly:

1. ArUco marker-based localization of the tested wooden cube; and
2. YOLO26 learning-based detection of the same tested wooden cube.

All Duck extensions, datasets, experiments, controllers, results, and narrative elements remain excluded. The word appears here only to document that deliberate scope exclusion. Raw YOLO images and labels, ChArUco capture images, per-trial camera images, caches, compiled files, environment directories, development failures, unrelated logs, and large media are also excluded.

## Dataset disclosure

The detector dataset contained 120 images with a deterministic split:

| Split | Images |
|---|---:|
| Train | 90 |
| Validation | 15 |
| Held-out test | 15 |

The raw 120 images and labels are not distributed. The split manifest, dataset configuration, saved training arguments, and training history are retained. The repository therefore supports provenance inspection but does not claim full detector-training or held-out-image reproducibility.

## Frozen official evidence

The public release retains 20 official ArUco trial JSON records and 20 official YOLO trial JSON records, for 40 physical benchmark records total. The included result artifacts remain byte-identical to the authoritative source.

| Metric | ArUco | YOLO |
|---|---:|---:|
| Harmonized successful full cycles | 20/20 | 20/20 |
| Mean cycle time | 133.489 s | 133.392 s |
| Median cycle time | 133.276 s | 133.170 s |
| Sample SD | 0.752 s | 0.623 s |
| Minimum | 132.599 s | 132.722 s |
| Maximum | 135.402 s | 134.551 s |

Harmonized image-space repeatability:

| Coordinate | Mean distance | Maximum distance |
|---|---:|---:|
| ArUco marker center | 1.269 px | 3.973 px |
| YOLO bounding-box center | 1.485 px | 5.026 px |
| YOLO upper-quarter grasp reference | 1.503 px | 5.057 px |

Held-out YOLO detector evidence remains separate from robot task success: 15 images, 15/15 ground-truth cubes detected, precision 0.916, recall 1.000, mAP@0.5 0.987, mAP@0.5:0.95 0.885, two additional background false positives, and zero false negatives.

These are bounded controlled results, not claims of universal reliability, generalization, or statistical superiority.

## Source-to-public provenance

The authoritative source root was accessed read-only:

    D:\Projects\Autonomous_Robotic_Grasping

Initial and final source baselines are identical:

| Property | Initial | Final |
|---|---:|---:|
| File count | 2,441 | 2,441 |
| Total bytes | 467,050,574 | 467,050,574 |
| Latest write time, UTC | 2026-08-29 11:31:51 | 2026-08-29 11:31:51 |

The provenance map contains 106 historical rows:

- 99 included_byte_identical rows, totaling 6,170,838 bytes;
- seven removed_public_safety_boundary rows;
- zero missing source files;
- zero missing included public files;
- zero unexpectedly present withheld files; and
- zero SHA-256 mismatches across the 99 included pairs.

The one renamed included artifact maps the source 06_pixel_to_robot\workspace_calibration_points_DAY07_FINAL_9PTS_BACKUP.json to mapping\final\workspace_calibration_points_day07_9pts.json; its contents remain byte-identical.

## Final repository inventory

- Files excluding .git: 125
- Total size excluding .git: 6,277,247 bytes
- Largest file: yolo/final/model/best.pt, 5,364,357 bytes
- Files larger than 10 MiB: 0
- Files larger than GitHub's 100 MiB single-file limit: 0
- Python files: 17

## Validation results

| Check | Result |
|---|---|
| Authoritative source baseline unchanged | PASS |
| Provenance-map schema and statuses | PASS, 99 included plus 7 safety-withheld |
| Source-to-public SHA-256 equality | PASS, 99/99 included pairs |
| Safety-withheld public paths absent | PASS, 7/7 |
| Executable actuation-pattern scan | PASS, none |
| Direct robot-motion instructions in documentation | PASS, none |
| Python py_compile using temporary output | PASS, 17/17 |
| Official physical trial JSON inventory | PASS, 20 ArUco plus 20 YOLO |
| README frozen-result assertions | PASS |
| Result artifacts unchanged from authoritative source | PASS |
| Markdown local-link targets | PASS, 19 Markdown files |
| CFF YAML parse and approved fields | PASS |
| Standard GNU AGPL v3 text and SHA-256 | PASS |
| SPDX license declarations | PASS, AGPL-3.0-only |
| Ultralytics/YOLO26 attribution and checkpoint notice | PASS |
| Raw 120-image dataset absent | PASS |
| Credential, token, email, and personal user-path scan | PASS, none |
| Invented DOI, arXiv, ORCID, or email identifiers | PASS, none |
| Unresolved license or author placeholders | PASS, none |
| Accidental Duck references outside this audit statement | PASS, none |
| Git whitespace check | PASS |
| Files larger than 10 MiB or 100 MiB | PASS, none |

## Publication boundaries

- The preprint is in preparation; no identifier is available.
- The raw detector dataset and held-out images are unavailable in this repository.
- Hardware calibration and pose records are specific to the archived experiment.
- The public release supports evidence inspection and software-side analysis, not physical robot operation.
- Publication itself remains a separate maintainer action.

## Blockers

None.

## Final determination

**PUBLICATION_READY**
