# api/vision.py
import os
from typing import Any, Dict, List

from PIL import Image
from ultralytics import YOLO

# Cached model instance
_MODEL = None

# You can override this with an env var in Render:
# NIBBLECHECK_YOLO_WEIGHTS=/path/to/nibblecheck-yolo.pt
_DEFAULT_WEIGHTS = os.getenv("NIBBLECHECK_YOLO_WEIGHTS", "yolov8n.pt")


def load_model(weights_path: str = None) -> None:
    """
    Load YOLO model into memory. Safe to call multiple times.
    """
    global _MODEL
    if _MODEL is not None:
        return

    path = weights_path or _DEFAULT_WEIGHTS
    _MODEL = YOLO(path)


def get_model():
    """
    Return the cached YOLO model, loading it on first use.
    """
    if _MODEL is None:
        load_model()
    return _MODEL


def detect_foods(image: Image.Image, conf_threshold: float = 0.25) -> List[Dict[str, Any]]:
    """
    Run object detection on a PIL image and return a simple list of detections:
    [{label, score, bbox}, ...]
    bbox is [x1, y1, x2, y2] in pixel coordinates.
    """
    model = get_model()

    if image.mode != "RGB":
        image = image.convert("RGB")

    results = model(image)[0]
    names = results.names if hasattr(results, "names") else model.names
    boxes = results.boxes

    detections: List[Dict[str, Any]] = []
    for cls_idx, conf, xyxy in zip(boxes.cls, boxes.conf, boxes.xyxy):
        score = float(conf)
        if score < conf_threshold:
            continue

        label = str(names[int(cls_idx)])
        bbox = [float(v) for v in xyxy.tolist()]

        detections.append(
            {
                "label": label,
                "score": score,
                "bbox": bbox,
            }
        )

    return detections
