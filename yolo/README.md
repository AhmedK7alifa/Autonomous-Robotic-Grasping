# YOLO controller

`controller.py` is the portable full-cycle entry point. Its functions are
unchanged from `benchmark_controller.py`; only support filenames and
repository-relative artifact paths differ.

```bash
python yolo/controller.py
```

The support chain is:

```text
controller.py
├── control_interface.py
│   └── extracts parity-checked motion functions from ArUco location_01.py
└── perception_mapping.py
    ├── perception.py
    └── mapping_interface.py
```

`perception.py` loads `model/best.pt` and the experiment's camera calibration.
`mapping_interface.py` loads the experimental mapping, source anchors, and
parity-checked ArUco mapping/correction functions. `benchmark_controller.py`
is the exact 24,039-byte controller used in the official experiment; its SHA-256 is
`8eb4fa72003b5af379aa443fdf6f74bfa72ab06d10cbb7ee5bca39d6ac380ac0`.

Run `python yolo/verify_equivalence.py` for static controller and perception
checks.
