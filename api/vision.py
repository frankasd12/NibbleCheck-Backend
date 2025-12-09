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


def detect_foods(image: Image.Image, conf_threshold: float = 0.25):
    model = get_model()

    if image.mode != "RGB":
        image = image.convert("RGB")

    # Downscale big images to speed up inference on CPU
    max_side = 480  # try 320–480 and see what feels good
    w, h = image.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        image = image.resize((int(w * scale), int(h * scale)))

    results = model(
        image,
        imgsz=max_side,
        conf=conf_threshold,
        verbose=False,
    )[0]

    names = results.names if hasattr(results, "names") else model.names
    boxes = results.boxes

    detections = []
    for cls_idx, conf, xyxy in zip(boxes.cls, boxes.conf, boxes.xyxy):
        score = float(conf)
        if score < conf_threshold:
            continue
        label = str(names[int(cls_idx)])
        bbox = [float(v) for v in xyxy.tolist()]
        detections.append({"label": label, "score": score, "bbox": bbox})

    return detections

