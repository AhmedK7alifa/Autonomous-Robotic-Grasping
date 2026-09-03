# Analysis

The analysis scripts read the 20 official records for each method,
recompute the reported statistics, and compare them with the committed summary
files. They do not modify benchmark records or result files.

```bash
python analysis/analyze_aruco.py
python analysis/analyze_yolo.py
python analysis/compare_methods.py
```

Cycle-time dispersion is the sample standard deviation. Harmonized
repeatability uses the arithmetic mean of the four observed coordinates at
each location as that method's empirical center. The separate ArUco
nominal-center metric uses the nominal centers stored in its trial records.
