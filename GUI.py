"""PyQt5 desktop client for image, folder, video and camera bicycle detection."""
import sys
from datetime import datetime
from pathlib import Path

import cv2
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QFileDialog, QFormLayout, QFrame, QHBoxLayout,
                             QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
                             QTextEdit, QVBoxLayout, QWidget)

from config import GlobalConfig
from detector import BicycleDetector


IMAGE_FILTER = "图片 (*.jpg *.jpeg *.png *.bmp)"
VIDEO_FILTER = "视频 (*.mp4 *.avi *.mov *.mkv)"


class VideoDetectThread(QThread):
    frame_signal = pyqtSignal(object, object)
    info_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, detector, source, save_path=None):
        super().__init__()
        self.detector = detector
        self.source = source
        self.save_path = Path(save_path) if save_path else None
        self.is_running = True

    def run(self):
        capture = cv2.VideoCapture(self.source)
        writer = None
        if not capture.isOpened():
            self.info_signal.emit("无法打开视频源，请检查文件、摄像头编号或流地址。")
            self.finished_signal.emit()
            return
        try:
            fps = capture.get(cv2.CAP_PROP_FPS) or GlobalConfig.VIDEO_FPS
            while self.is_running:
                ok, frame = capture.read()
                if not ok:
                    break
                annotated, detections = self.detector.predict(frame)
                if self.save_path:
                    if writer is None:
                        self.save_path.parent.mkdir(parents=True, exist_ok=True)
                        height, width = annotated.shape[:2]
                        writer = cv2.VideoWriter(str(self.save_path), cv2.VideoWriter_fourcc(*GlobalConfig.VIDEO_CODEC), fps, (width, height))
                    writer.write(annotated)
                self.frame_signal.emit(frame, annotated)
                self.info_signal.emit("检测到 {} 辆自行车".format(len(detections)))
                self.msleep(GlobalConfig.FRAME_DELAY_MS)
        except Exception as error:
            self.info_signal.emit(str(error))
        finally:
            capture.release()
            if writer:
                writer.release()
            self.finished_signal.emit()

    def stop(self):
        self.is_running = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        GlobalConfig.ensure_directories()
        self.detector = BicycleDetector()
        self.current_original = None
        self.current_result = None
        self.thread = None
        self.setWindowTitle(GlobalConfig.SYS_NAME)
        self.resize(GlobalConfig.WINDOW_WIDTH, GlobalConfig.WINDOW_HEIGHT)
        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        layout = QHBoxLayout(root)
        left = QFrame()
        left.setFixedWidth(GlobalConfig.LEFT_PANEL_WIDTH)
        controls = QVBoxLayout(left)
        controls.addWidget(QLabel("<h2>自行车检测</h2>"))
        form = QFormLayout()
        self.model_input = QLineEdit(str(GlobalConfig.MODEL_PATH))
        self.source_input = QLineEdit("0")
        form.addRow("模型权重", self.model_input)
        form.addRow("摄像头/流", self.source_input)
        controls.addLayout(form)
        for title, callback in (
            ("选择模型", self.choose_model), ("图片检测", self.detect_image),
            ("文件夹批量检测", self.detect_folder), ("视频检测", self.detect_video),
            ("摄像头实时检测", self.detect_camera), ("保存当前结果", self.save_current),
            ("停止视频/摄像头", self.stop_stream),
        ):
            button = QPushButton(title)
            button.clicked.connect(callback)
            controls.addWidget(button)
        controls.addWidget(QLabel("运行日志"))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        controls.addWidget(self.log, 1)

        preview = QVBoxLayout()
        labels = QHBoxLayout()
        labels.addWidget(QLabel("原始画面"), 1)
        labels.addWidget(QLabel("检测结果"), 1)
        preview.addLayout(labels)
        self.original_view = self._new_preview("等待输入")
        self.result_view = self._new_preview("等待检测")
        previews = QHBoxLayout()
        previews.addWidget(self.original_view, 1)
        previews.addWidget(self.result_view, 1)
        preview.addLayout(previews, 1)
        layout.addWidget(left)
        layout.addLayout(preview, 1)
        self.setCentralWidget(root)

    @staticmethod
    def _new_preview(message):
        label = QLabel(message)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("background:#20242a;color:#b8c1cc;border-radius:8px;")
        label.setMinimumSize(450, 500)
        return label

    def append_log(self, message):
        self.log.append("[{}] {}".format(datetime.now().strftime("%H:%M:%S"), message))

    def choose_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型", self.model_input.text(), "模型 (*.pt)")
        if path:
            self.model_input.setText(path)
            self.detector.set_model(path)
            self.append_log("模型已切换：{}".format(path))

    def _ensure_model(self):
        path = Path(self.model_input.text()).expanduser()
        if not path.exists():
            QMessageBox.warning(self, "未找到模型", "请选择有效的 .pt 模型权重。")
            return False
        self.detector.set_model(path)
        return True

    def detect_image(self):
        if not self._ensure_model():
            return
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", IMAGE_FILTER)
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            QMessageBox.warning(self, "读取失败", "无法读取图片文件。")
            return
        try:
            result, detections = self.detector.predict(frame)
            self.show_frames(frame, result)
            self.append_log("{}：检测到 {} 个目标".format(Path(path).name, len(detections)))
        except Exception as error:
            QMessageBox.critical(self, "检测失败", str(error))

    def detect_folder(self):
        if not self._ensure_model():
            return
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not folder:
            return
        image_paths = [path for path in Path(folder).iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        if not image_paths:
            QMessageBox.information(self, "没有图片", "该文件夹中没有支持的图片。")
            return
        output = GlobalConfig.OUTPUT_DIR / "folder_{}".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
        output.mkdir(parents=True, exist_ok=True)
        total = 0
        for index, path in enumerate(sorted(image_paths), 1):
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            result, detections = self.detector.predict(frame)
            cv2.imwrite(str(output / "{}_detected{}".format(path.stem, path.suffix)), result)
            self.show_frames(frame, result)
            total += len(detections)
            QApplication.processEvents()
            self.append_log("{}/{} {}：{} 个目标".format(index, len(image_paths), path.name, len(detections)))
        self.append_log("批量检测完成，{} 张图片，{} 个目标；结果：{}".format(len(image_paths), total, output))

    def detect_video(self):
        if not self._ensure_model():
            return
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", VIDEO_FILTER)
        if path:
            self.start_stream(path, "video_{}.mp4".format(datetime.now().strftime("%Y%m%d_%H%M%S")))

    def detect_camera(self):
        if not self._ensure_model():
            return
        source = self.source_input.text().strip() or "0"
        source = int(source) if source.isdigit() else source
        self.start_stream(source, "camera_{}.mp4".format(datetime.now().strftime("%Y%m%d_%H%M%S")))

    def start_stream(self, source, filename):
        self.stop_stream()
        path = GlobalConfig.OUTPUT_DIR / filename
        self.thread = VideoDetectThread(self.detector, source, path)
        self.thread.frame_signal.connect(self.show_frames)
        self.thread.info_signal.connect(self.append_log)
        self.thread.finished_signal.connect(lambda: self.append_log("视频任务已停止，结果：{}".format(path)))
        self.thread.start()

    def stop_stream(self):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait(3000)
        self.thread = None

    def show_frames(self, original, result):
        self.current_original, self.current_result = original.copy(), result.copy()
        self.original_view.setPixmap(self._to_pixmap(original))
        self.result_view.setPixmap(self._to_pixmap(result))

    @staticmethod
    def _to_pixmap(frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
        return QPixmap.fromImage(image).scaled(520, 540, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def save_current(self):
        if self.current_result is None:
            QMessageBox.information(self, "没有结果", "请先完成一次图片或视频检测。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存检测结果", str(GlobalConfig.OUTPUT_DIR / "detected.jpg"), "图片 (*.jpg *.png)")
        if path and cv2.imwrite(path, self.current_result):
            self.append_log("当前结果已保存：{}".format(path))

    def closeEvent(self, event):
        self.stop_stream()
        event.accept()


if __name__ == "__main__":
    application = QApplication(sys.argv)
    application.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(application.exec_())
