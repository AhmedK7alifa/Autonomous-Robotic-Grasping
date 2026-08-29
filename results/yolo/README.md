# YOLO benchmark records

The yolo_final_benchmark directory contains four JSON records for each of five locations plus the official summary files. No per-trial images were part of the frozen YOLO benchmark record set.

Run 11_analyze_final_yolo_benchmark.py from this directory to regenerate the method summary. The promoter is preserved to document how validated records were admitted to the official set; it should not be rerun against frozen results.

Official result: 20/20 observed successful full cycles under the controlled benchmark conditions; mean cycle time 133.392 s; bounding-box-center repeatability mean 1.485 px; grasp-reference repeatability mean 1.503 px.
