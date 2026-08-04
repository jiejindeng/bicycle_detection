"""Run a quick validation prediction on a test image."""
from pathlib import Path

from ultralytics import YOLO

from config import GlobalConfig


def main():
    model_path = GlobalConfig.BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
    if not model_path.exists():
        model_path = GlobalConfig.MODEL_PATH
    source = Path("test.jpg")
    if not source.exists():
        raise FileNotFoundError("请准备测试图片 test.jpg，或自行修改 val.py 中的 source。")
    results = YOLO(str(model_path))(str(source))
    print("检测到", len(results[0].boxes), "个目标")
    results[0].save(filename="result.jpg")


if __name__ == "__main__":
    main()
