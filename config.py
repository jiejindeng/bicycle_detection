from pathlib import Path


class GlobalConfig:
    """Application-wide defaults for bicycle detection."""

    SYS_NAME = "基于YOLO26的自行车检测系统"
    BASE_DIR = Path(__file__).resolve().parent
    MODEL_PATH = BASE_DIR / "models" / "best.pt"
    CONF_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45
    CLASS_NAMES = {0: "bicycle"}
    CAMERA_DEVICE_ID = 0
    FRAME_DELAY_MS = 30
    WINDOW_WIDTH = 1400
    WINDOW_HEIGHT = 700
    LEFT_PANEL_WIDTH = 320
    RIGHT_PANEL_WIDTH = 1040
    VIDEO_CODEC = "mp4v"
    VIDEO_FPS = 20.0
    OUTPUT_DIR = BASE_DIR / "output"

    @classmethod
    def ensure_directories(cls):
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
