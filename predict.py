"""Predict one image and save the Ultralytics annotated result."""
from pathlib import Path

from ultralytics import YOLO

from config import GlobalConfig


def main():
    source = Path("test.jpg")
    if not source.exists():
        raise FileNotFoundError("请准备待检测图片 test.jpg，或自行修改 predict.py 中的 source。")
    model_path = GlobalConfig.BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
    model = YOLO(str(model_path if model_path.exists() else GlobalConfig.MODEL_PATH))
    model.predict(str(source), save=True, conf=GlobalConfig.CONF_THRESHOLD, iou=GlobalConfig.IOU_THRESHOLD)


if __name__ == "__main__":
    main()
