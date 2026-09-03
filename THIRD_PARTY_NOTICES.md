# Third-party notices

This repository contains original project materials by AHMED KHALIFA IDRISS ELHAJ and uses third-party software and model technology. Third-party components remain the property of their respective copyright holders and are subject to their applicable licenses.

## Ultralytics YOLO26

This project uses **Ultralytics YOLO26** for the learning-based cube detector. YOLO26, the Ultralytics software framework, underlying architecture, pretrained components, code, models, and documentation were created and are maintained by **Ultralytics and its contributors**. The project author does not claim authorship or ownership of the underlying YOLO26 architecture or toolchain.

Ultralytics documents YOLO26 code, models, and documentation as available under **AGPL-3.0** and Enterprise licensing options:

- Ultralytics YOLO26 documentation: https://docs.ultralytics.com/models/yolo26/
- Ultralytics repository: https://github.com/ultralytics/ultralytics
- Ultralytics licensing information: https://www.ultralytics.com/license

This repository is licensed under **AGPL-3.0-only**. The unmodified GNU AGPLv3 license text is included in [LICENSE](LICENSE).

## Project-specific checkpoint

[yolo/model/best.pt](yolo/model/best.pt) is the project-specific trained YOLO26 checkpoint produced for the controlled wooden-cube detector. Its inclusion does not transfer or imply ownership of the YOLO26 architecture, Ultralytics framework, or pretrained-model lineage.

## Other dependencies

The project also depends on packages including OpenCV, NumPy, PyTorch, TorchVision, LeRobot, the Feetech Servo SDK, PySerial, and PyYAML. Those packages are not vendored here and remain governed by their respective upstream licenses. Their appearance in requirements.txt or environment.yml is dependency documentation, not a relicensing of those projects.
