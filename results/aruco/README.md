# ArUco benchmark records

The aruco_final_benchmark_v3 directory contains four JSON records for each of five locations plus the official summary files. The 20 corresponding camera images were excluded.

The analyzer intentionally remains byte-identical to the experimental version and expects 06_pixel_to_robot/aruco_final_benchmark_v3 beneath the process working directory. To regenerate the method summary, place this directory’s contents under 06_pixel_to_robot in a disposable staging tree, run the script from the staging root, and compare the generated metrics with the committed outputs.

Official result: 20/20 observed successful full cycles under the controlled benchmark conditions; mean cycle time 133.489 s; harmonized marker-center repeatability mean 1.269 px.
