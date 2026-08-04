"""Train the bicycle detector using the Ultralytics CLI API."""
import warnings

from torch.xpu import device
from ultralytics import YOLO

warnings.filterwarnings("ignore")


def main():
    model = YOLO("yolo26n.pt")
    model.train(
        data="VOCData/mydata.yaml",
        epochs=100,
        batch=32,
        imgsz=640,
        workers=8,
        device="mps",
        optimizer="SGD",
        amp=False,
        cache=False,
    )


if __name__ == "__main__":
    main()
