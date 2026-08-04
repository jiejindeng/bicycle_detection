from __future__ import annotations

import sys
from typing import Optional, Union

import cv2
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
                             QMessageBox, QPushButton, QVBoxLayout, QWidget)

from config import GlobalConfig
from detector import BicycleDetector


class VideoDetectThread(QThread):
    frame_signal = pyqtSignal(object)
    info_signal = pyqtSignal(str)

    def __init__(self, source: Union[str, int]):
        super().__init__()
        self.source = source
        self.running = True

    def run(self):
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            self.info_signal.emit("无法打开视频源")
            return
        try:
            detector = BicycleDetector()
            while self.running:
                ok, frame = capture.read()
                if not ok:
                    break
                annotated, items = detector.predict(frame)
                self.frame_signal.emit(annotated)
                self.info_signal.emit(f"当前检测到 {len(items)} 辆自行车")
                self.msleep(GlobalConfig.FRAME_DELAY_MS)
        except Exception as error:
            self.info_signal.emit(str(error))
        finally:
            capture.release()

    def stop(self):
        self.running = False
        self.wait(2000)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.thread: Optional[VideoDetectThread] = None
        self.setWindowTitle(GlobalConfig.SYS_NAME)
        self.resize(1000, 700)

        self.preview = QLabel("请选择图片、视频或摄像头")
        self.preview.setMinimumSize(800, 520)
        self.preview.setStyleSheet("background:#1f2329;color:#b8c1cc;font-size:20px;")
        self.preview.setScaledContents(True)
        self.status = QLabel("就绪")
        image_button = QPushButton("检测图片")
        video_button = QPushButton("检测视频")
        camera_button = QPushButton("打开摄像头")
        stop_button = QPushButton("停止")
        image_button.clicked.connect(self.detect_image)
        video_button.clicked.connect(self.detect_video)
        camera_button.clicked.connect(lambda: self.start_stream(GlobalConfig.CAMERA_DEVICE_ID))
        stop_button.clicked.connect(self.stop_stream)

        buttons = QHBoxLayout()
        for button in (image_button, video_button, camera_button, stop_button):
            buttons.addWidget(button)
        layout = QVBoxLayout()
        layout.addWidget(self.preview)
        layout.addLayout(buttons)
        layout.addWidget(self.status)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def detect_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片 (*.jpg *.jpeg *.png *.bmp)")
        if not path:
            return
        try:
            frame = cv2.imread(path)
            annotated, items = BicycleDetector().predict(frame)
            self.show_frame(annotated)
            self.status.setText(f"检测完成：{len(items)} 个目标")
        except Exception as error:
            QMessageBox.critical(self, "检测失败", str(error))

    def detect_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频 (*.mp4 *.avi *.mov *.mkv)")
        if path:
            self.start_stream(path)

    def start_stream(self, source):
        self.stop_stream()
        self.thread = VideoDetectThread(source)
        self.thread.frame_signal.connect(self.show_frame)
        self.thread.info_signal.connect(self.status.setText)
        self.thread.start()

    def stop_stream(self):
        if self.thread:
            self.thread.stop()
            self.thread = None

    def show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888)
        self.preview.setPixmap(QPixmap.fromImage(image))

    def closeEvent(self, event):
        self.stop_stream()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
