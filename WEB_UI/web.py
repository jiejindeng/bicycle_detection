"""Flask web interface for bicycle image inference."""
import base64
import sys
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import GlobalConfig
from detector import BicycleDetector

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
detector = BicycleDetector()


def infer(image_bytes):
    frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("文件不是有效图片。")
    annotated, detections = detector.predict(frame)
    ok, buffer = cv2.imencode(".jpg", annotated)
    if not ok:
        raise RuntimeError("无法编码检测结果。")
    return base64.b64encode(buffer.tobytes()).decode("ascii"), detector.detection_rows(detections)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    rows = []
    error = None
    if request.method == "POST":
        uploaded = request.files.get("image")
        if not uploaded or not uploaded.filename:
            error = "请选择图片文件。"
        else:
            try:
                result, rows = infer(uploaded.read())
            except Exception as exc:
                error = str(exc)
    return render_template("index.html", result=result, rows=rows, error=error)


@app.post("/api/predict")
def api_predict():
    uploaded = request.files.get("image")
    if not uploaded:
        return jsonify({"error": "缺少 image 文件字段"}), 400
    try:
        result, rows = infer(uploaded.read())
        return jsonify({"count": len(rows), "detections": rows, "image": result})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    GlobalConfig.ensure_directories()
    app.run(host="0.0.0.0", port=5001, debug=True)
