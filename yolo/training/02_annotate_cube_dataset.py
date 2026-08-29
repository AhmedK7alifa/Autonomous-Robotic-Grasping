"""Manual one-box YOLO annotation tool for the wooden-cube dataset.

Controls:
    Left mouse drag: draw one bounding box
    S: save the current box and move to the next image
    N: move to the next image without changing its label
    B: move back to the previous image
    R: clear the draft box; explicitly unlock an existing label for redo
    Q or Esc: quit without changing any unsaved annotation
"""

from pathlib import Path
import math

import cv2


CLASS_ID = 0
WINDOW_NAME = "Cube Dataset Annotation"

SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = SCRIPT_DIR / "dataset_raw" / "images"
LABEL_DIR = SCRIPT_DIR / "dataset_raw" / "labels"

Box = tuple[int, int, int, int]
YoloBox = tuple[float, float, float, float]


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def pixel_box_to_yolo(
    box: Box,
    image_width: int,
    image_height: int,
) -> YoloBox:
    """Convert an x1/y1/x2/y2 pixel box to normalized YOLO values."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")

    x1, y1, x2, y2 = box
    x1, x2 = sorted((clamp(x1, 0, image_width - 1), clamp(x2, 0, image_width - 1)))
    y1, y2 = sorted(
        (clamp(y1, 0, image_height - 1), clamp(y2, 0, image_height - 1))
    )

    if x2 <= x1 or y2 <= y1:
        raise ValueError("The bounding box must have positive width and height.")

    x_center = ((x1 + x2) / 2.0) / image_width
    y_center = ((y1 + y2) / 2.0) / image_height
    box_width = (x2 - x1) / image_width
    box_height = (y2 - y1) / image_height

    normalized = (x_center, y_center, box_width, box_height)

    if not all(math.isfinite(value) for value in normalized):
        raise ValueError("The normalized coordinates must be finite.")

    if not all(0.0 <= value <= 1.0 for value in normalized):
        raise ValueError("Every normalized coordinate must be within [0, 1].")

    return normalized


def yolo_box_to_pixel(
    normalized: YoloBox,
    image_width: int,
    image_height: int,
) -> Box:
    """Convert a normalized YOLO box back to display pixel coordinates."""
    x_center, y_center, box_width, box_height = normalized

    if not all(math.isfinite(value) for value in normalized):
        raise ValueError("Existing label contains a non-finite coordinate.")

    if not all(0.0 <= value <= 1.0 for value in normalized):
        raise ValueError("Existing label contains a coordinate outside [0, 1].")

    if box_width <= 0.0 or box_height <= 0.0:
        raise ValueError("Existing label has a non-positive box size.")

    x1_normalized = x_center - box_width / 2.0
    y1_normalized = y_center - box_height / 2.0
    x2_normalized = x_center + box_width / 2.0
    y2_normalized = y_center + box_height / 2.0

    corners = (
        x1_normalized,
        y1_normalized,
        x2_normalized,
        y2_normalized,
    )
    tolerance = 1e-6
    if not all(-tolerance <= value <= 1.0 + tolerance for value in corners):
        raise ValueError("Existing label extends outside the image.")

    x1 = clamp(round(x1_normalized * image_width), 0, image_width - 1)
    y1 = clamp(round(y1_normalized * image_height), 0, image_height - 1)
    x2 = clamp(round(x2_normalized * image_width), 0, image_width - 1)
    y2 = clamp(round(y2_normalized * image_height), 0, image_height - 1)

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Existing label produces an empty pixel box.")

    return x1, y1, x2, y2


def label_path_for(image_path: Path) -> Path:
    return LABEL_DIR / f"{image_path.stem}.txt"


def load_existing_box(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> Box | None:
    if not label_path.exists():
        return None

    lines = [
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) != 1:
        raise ValueError(
            f"Expected exactly one annotation in {label_path}, found {len(lines)}."
        )

    fields = lines[0].split()
    if len(fields) != 5:
        raise ValueError(f"Invalid YOLO label format in {label_path}.")

    if fields[0] != str(CLASS_ID):
        raise ValueError(
            f"Expected class {CLASS_ID} in {label_path}, found {fields[0]}."
        )

    try:
        normalized = tuple(float(value) for value in fields[1:])
    except ValueError as error:
        raise ValueError(f"Non-numeric coordinate in {label_path}.") from error

    return yolo_box_to_pixel(normalized, image_width, image_height)


def save_yolo_label(
    label_path: Path,
    normalized: YoloBox,
    allow_overwrite: bool,
) -> None:
    if not all(0.0 <= value <= 1.0 for value in normalized):
        raise ValueError("Refusing to save a coordinate outside [0, 1].")

    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_line = (
        f"{CLASS_ID} "
        f"{normalized[0]:.6f} "
        f"{normalized[1]:.6f} "
        f"{normalized[2]:.6f} "
        f"{normalized[3]:.6f}\n"
    )

    if label_path.exists():
        if not allow_overwrite:
            raise FileExistsError(
                f"Existing annotation is protected: {label_path}. Press R first."
            )
        label_path.write_text(label_line, encoding="utf-8")
    else:
        with label_path.open("x", encoding="utf-8") as label_file:
            label_file.write(label_line)


class AnnotationTool:
    def __init__(self, image_paths: list[Path]) -> None:
        self.image_paths = image_paths
        self.index = self.find_resume_index()
        self.image = None
        self.image_width = 0
        self.image_height = 0
        self.saved_box: Box | None = None
        self.draft_box: Box | None = None
        self.drag_start: tuple[int, int] | None = None
        self.drawing = False
        self.replacement_armed = False
        self.message = ""
        self.load_current_image()

    def find_resume_index(self) -> int:
        for index, image_path in enumerate(self.image_paths):
            if not label_path_for(image_path).exists():
                return index
        return 0

    @property
    def current_image_path(self) -> Path:
        return self.image_paths[self.index]

    @property
    def current_label_path(self) -> Path:
        return label_path_for(self.current_image_path)

    def completed_count(self) -> int:
        return sum(
            1
            for image_path in self.image_paths
            if label_path_for(image_path).is_file()
        )

    def load_current_image(self) -> None:
        image = cv2.imread(str(self.current_image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {self.current_image_path}")

        self.image = image
        self.image_height, self.image_width = image.shape[:2]
        self.saved_box = load_existing_box(
            self.current_label_path,
            self.image_width,
            self.image_height,
        )
        self.draft_box = None
        self.drag_start = None
        self.drawing = False
        self.replacement_armed = False

        if self.saved_box is None:
            self.message = "Draw a tight cube box, then press S."
        else:
            self.message = "Existing label protected. Press R to redo it."

    def set_message(self, message: str) -> None:
        self.message = message
        print(message)

    def mouse_callback(self, event, x, y, _flags, _parameter) -> None:
        x = clamp(x, 0, self.image_width - 1)
        y = clamp(y, 0, self.image_height - 1)

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.saved_box is not None and not self.replacement_armed:
                self.set_message("Label protected: press R before drawing a replacement.")
                return

            self.drag_start = (x, y)
            self.draft_box = None
            self.drawing = True

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            start_x, start_y = self.drag_start
            self.draft_box = (
                min(start_x, x),
                min(start_y, y),
                max(start_x, x),
                max(start_y, y),
            )

        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            start_x, start_y = self.drag_start
            candidate = (
                min(start_x, x),
                min(start_y, y),
                max(start_x, x),
                max(start_y, y),
            )

            try:
                pixel_box_to_yolo(
                    candidate,
                    self.image_width,
                    self.image_height,
                )
            except ValueError:
                self.draft_box = None
                self.set_message("Box is empty. Drag a box with positive size.")
            else:
                self.draft_box = candidate
                self.set_message("Draft box ready. Press S to save or R to redraw.")

    def draw_display(self):
        display = self.image.copy()

        if self.saved_box is not None:
            saved_color = (128, 128, 128) if self.replacement_armed else (0, 255, 0)
            x1, y1, x2, y2 = self.saved_box
            cv2.rectangle(display, (x1, y1), (x2, y2), saved_color, 2)

        if self.draft_box is not None:
            x1, y1, x2, y2 = self.draft_box
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)

        return display

    def update_window_title(self) -> None:
        title = (
            f"Cube Annotation - Image {self.index + 1} / {len(self.image_paths)}"
            f" - Completed {self.completed_count()} / {len(self.image_paths)}"
            f" - {self.current_image_path.name} - {self.message}"
        )
        cv2.setWindowTitle(WINDOW_NAME, title)

    def move(self, offset: int) -> None:
        if self.draft_box is not None:
            self.set_message("Unsaved draft: press S to save or R to clear it first.")
            return

        new_index = self.index + offset
        if not 0 <= new_index < len(self.image_paths):
            self.set_message("Already at the first or last image.")
            return

        self.index = new_index
        self.load_current_image()

    def redo(self) -> None:
        self.draft_box = None
        self.drawing = False
        self.drag_start = None

        if self.saved_box is not None:
            self.replacement_armed = True
            self.set_message("Redo enabled. Draw a replacement box, then press S.")
        else:
            self.set_message("Draft cleared. Draw a new cube box.")

    def save_and_advance(self) -> None:
        if self.draft_box is None:
            self.set_message("No draft box to save. Draw the cube box first.")
            return

        replacing_existing = self.current_label_path.exists()
        if replacing_existing and not self.replacement_armed:
            self.set_message("Existing label protected: press R before replacing it.")
            return

        normalized = pixel_box_to_yolo(
            self.draft_box,
            self.image_width,
            self.image_height,
        )
        save_yolo_label(
            self.current_label_path,
            normalized,
            allow_overwrite=self.replacement_armed,
        )
        print(f"Saved: {self.current_label_path}")

        if self.index + 1 < len(self.image_paths):
            self.index += 1
            self.load_current_image()
        else:
            self.load_current_image()
            self.set_message("Last image saved. Annotation set is complete.")

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)

        print(f"Images: {IMAGE_DIR}")
        print(f"Labels: {LABEL_DIR}")
        print(f"Found {len(self.image_paths)} images.")
        print(
            "Controls: mouse drag=draw, S=save+next, N=next, "
            "B=previous, R=redo/unlock, Q/Esc=quit"
        )

        try:
            while True:
                cv2.imshow(WINDOW_NAME, self.draw_display())
                self.update_window_title()
                key = cv2.waitKey(20) & 0xFF

                if key in (ord("q"), ord("Q"), 27):
                    if self.draft_box is not None:
                        print("Unsaved draft was not written.")
                    break
                if key in (ord("s"), ord("S")):
                    self.save_and_advance()
                elif key in (ord("n"), ord("N")):
                    self.move(1)
                elif key in (ord("b"), ord("B")):
                    self.move(-1)
                elif key in (ord("r"), ord("R")):
                    self.redo()
        finally:
            cv2.destroyAllWindows()
            print("Annotation tool closed safely.")


def main() -> None:
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

    image_paths = sorted(IMAGE_DIR.glob("cube_*.jpg"))
    if not image_paths:
        raise RuntimeError(f"No cube JPEG images found in: {IMAGE_DIR}")

    AnnotationTool(image_paths).run()


if __name__ == "__main__":
    main()
