# ArUco controller

`controller.py` is the portable full-cycle entry point:

```bash
python aruco/controller.py --location 1
```

`locations.yaml` contains only the five output/record metadata sets. Perception,
mapping, source-anchor correction, right-corridor behavior, motion constants,
contact detection, retention, placement, and release logic remain common.

The exact files used for the official benchmark are in
`benchmark_controllers`. They retain their original source bytes under shorter
filesystem names:

| File | SHA-256 |
|---|---|
| `location_01.py` | `209cbe4b0def1eb2b569475dc3f9e3a4aad81bd7bc56617a1f9f056c607966ac` |
| `location_02.py` | `582b178f6ebd8d8e7354afc59c7dac27d7fdbcbfa8292ceddcf2165baa010710` |
| `location_03.py` | `8ecf4588939e655d8cf99534e39ec4d0af2f63b25edafe459b1085b9026b1358` |
| `location_04.py` | `462d99a1727aefd7aba4604c69030edefc90ffbe10dce6c0c5bc29d3d750c4a4` |
| `location_05.py` | `92a942b2a12c6d6c40f4cf34d1c0e38fa1cb75aa8af6bccd478d4adc6aa4d3b6` |

Run `python aruco/verify_equivalence.py` for static hash, function-AST,
constant, and metadata checks against these five benchmark controller files.
