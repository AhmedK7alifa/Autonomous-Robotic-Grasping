# Results and analysis

This directory contains the publication-safe official records for the 40 physical benchmark trials.

- [aruco](aruco): 20 ArUco trial JSON files, official TXT/JSON/CSV summaries, and the frozen analyzer.
- [yolo](yolo): 20 YOLO trial JSON files, official TXT/JSON/CSV summaries, the record promoter, and the frozen analyzer.
- [comparison](comparison): frozen cross-method analysis script and official TXT/JSON/CSV comparison outputs.
- [figures](figures): publication-safe mapping visualization.

Per-trial camera images are excluded. The machine-readable JSON trial records and detailed CSV summaries remain sufficient to audit the reported success counts, cycle-time statistics, and localization-repeatability calculations.

The preserved analyzers retain their original path assumptions. The YOLO analyzer resolves data relative to its script and can run from its current directory. The ArUco analyzer resolves 06_pixel_to_robot/aruco_final_benchmark_v3 from the process working directory, and the comparison analyzer also expects the original numbered project topology. Both are included unchanged for provenance; use a separate staging copy when regenerating their outputs.

The output JSON/CSV files contain historical D:\Projects\... source paths. These record the experimental filesystem location and do not identify a person or contain credentials.
