from __future__ import annotations

import base64

import cv2
import numpy as np
from flask import Flask, render_template, request

from config import GlobalConfig
from detector import BicycleDetector

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
detector = BicycleDetector()


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    count = None
    if request.method == "POST":
        uploaded = request.files.get("image")
        if not uploaded or not uploaded.filename:
            error = "请选择一张图片。"
        else:
            image_data = uploaded.read()
            frame = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                error = "文件不是有效图片。"
            else:
                try:
                    annotated, detections = detector.predict(frame)
                    ok, encoded = cv2.imencode(".jpg", annotated)
                    result = base64.b64encode(encoded.tobytes()).decode() if ok else None
                    count = len(detections)
                except Exception as exc:
                    error = str(exc)
    return render_template("index.html", result=result, error=error, count=count)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
