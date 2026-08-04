from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2

from config import GlobalConfig


@dataclass(frozen=True)
class Detection:
    class_id: int
    label: str
    confidence: float
    xyxy: tuple[int, int, int, int]


class BicycleDetector:
    """Thin, testable wrapper around an Ultralytics YOLO model."""

    def __init__(self, model_path=GlobalConfig.MODEL_PATH):
        self.model_path = Path(model_path)
        self._model = None

    def _load_model(self):
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"未找到模型权重：{self.model_path}。请训练后将 best.pt 放入 weights/。"
                )
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path))
        return self._model

    def set_model(self, model_path):
        """Switch model weights and release the cached model."""
        self.model_path = Path(model_path)
        self._model = None

    def predict(self, frame, confidence: float = GlobalConfig.CONF_THRESHOLD,
                iou: float = GlobalConfig.IOU_THRESHOLD,
                classes: Optional[Iterable[int]] = None) -> Tuple[object, list[Detection]]:
        model = self._load_model()
        results = model(frame, conf=confidence, iou=iou, classes=list(classes) if classes else None,
                        verbose=False)
        result = results[0]
        detections: list[Detection] = []
        if result.boxes is None:
            return frame.copy(), detections

        names = result.names
        for box, class_id, score in zip(result.boxes.xyxy.cpu().tolist(),
                                        result.boxes.cls.cpu().tolist(),
                                        result.boxes.conf.cpu().tolist()):
            category = int(class_id)
            label = str(names.get(category, GlobalConfig.CLASS_NAMES.get(category, category)))
            x1, y1, x2, y2 = (round(value) for value in box)
            detections.append(Detection(category, label, float(score), (x1, y1, x2, y2)))

        return self.draw(frame.copy(), detections), detections

    @staticmethod
    def detection_rows(detections):
        return [
            {
                "class_id": item.class_id,
                "label": item.label,
                "confidence": round(item.confidence, 4),
                "x1": item.xyxy[0], "y1": item.xyxy[1],
                "x2": item.xyxy[2], "y2": item.xyxy[3],
            }
            for item in detections
        ]

    @staticmethod
    def draw(frame, detections: list[Detection]):
        for item in detections:
            x1, y1, x2, y2 = item.xyxy
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 170, 255), 2)
            text = f"{item.label} {item.confidence:.2f}"
            cv2.putText(frame, text, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 170, 255), 2, cv2.LINE_AA)
        return frame
