import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QLabel, QComboBox, QDialog
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, QTimer, QPoint, QRect

class ShapeDrawingDialog(QDialog):
    def __init__(self, parent=None, background_image=None):
        super().__init__(parent)
        self.setWindowTitle('도형 그리기 및 비디오 제어')
        self.setGeometry(200, 200, 800, 600)
        self.background_image = background_image
        self.shape_rect = QRect()
        self.drawing = False
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.shape_type = "원"
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.frames = []
        self.is_recording = False
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        control_layout = QHBoxLayout()

        self.shape_combo = QComboBox(self)
        self.shape_combo.addItem("원")
        self.shape_combo.addItem("사각형")
        control_layout.addWidget(self.shape_combo)

        self.effect_combo = QComboBox(self)
        self.effect_combo.addItem("인상주의")
        self.effect_combo.addItem("표현주의")
        control_layout.addWidget(self.effect_combo)

        self.btn_start_video = QPushButton('비디오시작', self)
        self.btn_start_video.clicked.connect(self.toggle_video)
        control_layout.addWidget(self.btn_start_video)

        self.btn_save_video = QPushButton('비디오저장', self)
        self.btn_save_video.clicked.connect(self.save_video)
        self.btn_save_video.setEnabled(False)
        control_layout.addWidget(self.btn_save_video)

        layout.addLayout(control_layout)

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(640, 480)
        layout.addWidget(self.image_label)

        self.setLayout(layout)
        self.update_image()
        self.setStyleSheet("""
            QDialog {
                background-color: #f0f0f0;
            }
            QComboBox {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 5px;
                min-width: 6em;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 15px;
                border-left-width: 1px;
                border-left-color: #cccccc;
                border-left-style: solid;
            }
        """)
    def update_frame(self):
        if self.shape_rect.isEmpty() or not self.cap or self.background_image is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)

        # 마스크 생성
        mask = np.zeros((480, 640), dtype=np.uint8)
        if self.shape_type == "원":
            center = ((self.shape_rect.left() + self.shape_rect.right()) // 2,
                      (self.shape_rect.top() + self.shape_rect.bottom()) // 2)
            axes = (self.shape_rect.width() // 2, self.shape_rect.height() // 2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        else:
            cv2.rectangle(mask, (self.shape_rect.left(), self.shape_rect.top()),
                          (self.shape_rect.right(), self.shape_rect.bottom()), 255, -1)

        frame_masked = cv2.bitwise_and(frame, frame, mask=mask)

        if self.effect_combo.currentText() == "인상주의":
            frame_effect = cv2.xphoto.oilPainting(frame_masked, 7, 1)
        else:  # "표현주의"
            frame_effect = self.apply_expressionist_effect(frame_masked)

        background_masked = cv2.bitwise_and(self.background_image, self.background_image, mask=cv2.bitwise_not(mask))
        final_frame = cv2.add(background_masked, frame_effect)

        if self.is_recording:
            self.frames.append(final_frame.copy())  # 프레임의 복사본 저장

        height, width, channel = final_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(final_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.image_label.underMouse():
            self.drawing = True
            self.start_point = event.pos() - self.image_label.pos()
            self.end_point = self.start_point
            self.shape_type = self.shape_combo.currentText()

    def mouseMoveEvent(self, event):
        if self.drawing and self.image_label.underMouse():
            self.end_point = event.pos() - self.image_label.pos()
            self.shape_rect = QRect(self.start_point, self.end_point).normalized()
            self.update_image()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            self.end_point = event.pos() - self.image_label.pos()
            self.shape_rect = QRect(self.start_point, self.end_point).normalized()
            self.update_image()

    def toggle_video(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_start_video.setText('비디오시작')
            if self.cap:
                self.cap.release()
                self.cap = None
            self.is_recording = False
            self.btn_save_video.setEnabled(True)
        else:
            self.btn_start_video.setText('비디오멈춤')
            self.frames = []
            self.is_recording = True
            self.btn_save_video.setEnabled(False)

            if not self.cap:
                self.cap = cv2.VideoCapture(0)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.timer.start(30)

    def update_frame(self):
        if self.shape_rect.isEmpty() or not self.cap or self.background_image is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)

        mask = np.zeros((480, 640), dtype=np.uint8)
        if self.shape_type == "원":
            center = ((self.shape_rect.left() + self.shape_rect.right()) // 2,
                      (self.shape_rect.top() + self.shape_rect.bottom()) // 2)
            axes = (self.shape_rect.width() // 2, self.shape_rect.height() // 2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        else:
            cv2.rectangle(mask, (self.shape_rect.left(), self.shape_rect.top()),
                          (self.shape_rect.right(), self.shape_rect.bottom()), 255, -1)

        frame_masked = cv2.bitwise_and(frame, frame, mask=mask)

        if self.effect_combo.currentText() == "인상주의":
            frame_effect = cv2.xphoto.oilPainting(frame_masked, 7, 1)
        else:  # "표현주의"
            frame_effect = self.apply_expressionist_effect(frame_masked)

        background_masked = cv2.bitwise_and(self.background_image, self.background_image, mask=cv2.bitwise_not(mask))
        final_frame = cv2.add(background_masked, frame_effect)

        if self.is_recording:
            self.frames.append(final_frame)

        height, width, channel = final_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(final_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)

    def apply_expressionist_effect(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.add(s, 50)
        v = cv2.add(v, 30)
        hsv_enhanced = cv2.merge([h, s, v])
        enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges = cv2.dilate(edges, None)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        expressionist = cv2.addWeighted(enhanced, 0.7, edges, 0.3, 0)

        kernel = np.ones((5, 5), np.float32) / 25
        expressionist = cv2.filter2D(expressionist, -1, kernel)

        return expressionist

    def save_video(self):
        if not self.frames:
            return

        file_dialog = QFileDialog()
        video_path, _ = file_dialog.getSaveFileName(self, '비디오 저장', '', 'Video Files (*.avi)')

        if video_path:
            # XVID 코덱 사용
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))

            try:
                for frame in self.frames:
                    # RGB to BGR 변환
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
            except Exception as e:
                print(f"비디오 저장 중 오류 발생: {e}")
            finally:
                out.release()
                print(f"비디오가 다음 경로에 저장되었습니다: {video_path}")




class VideoEffectApp(QWidget):
    def __init__(self):
        super().__init__()
        self.target_size = (640, 480)
        self.initUI()

        self.image_path = None
        self.background_image = None

    def initUI(self):
        self.setWindowTitle('인터렉티브동기유발')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        btn_layout = QHBoxLayout()
        self.btn_select_image = QPushButton('이미지선택', self)
        self.btn_select_image.clicked.connect(self.select_image)
        btn_layout.addWidget(self.btn_select_image)

        self.btn_set_shape = QPushButton('도형그리기', self)
        self.btn_set_shape.clicked.connect(self.open_shape_drawing)
        btn_layout.addWidget(self.btn_set_shape)

        layout.addLayout(btn_layout)

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(*self.target_size)
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        self.setLayout(layout)

        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                font-family: Arial, sans-serif;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                font-size: 14px;
                margin: 4px 2px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QComboBox {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 5px;
                min-width: 6em;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 15px;
                border-left-width: 1px;
                border-left-color: #cccccc;
                border-left-style: solid;
            }
            QLabel {
                background-color: #ffffff;
                border: 1px solid #cccccc;
            }
        """)


    def select_image(self):
        file_dialog = QFileDialog()
        self.image_path, _ = file_dialog.getOpenFileName(self, '이미지 선택', '', 'Image Files (*.png *.jpg *.bmp)')
        if self.image_path:
            # PNG 파일 처리 개선
            image = cv2.imdecode(np.fromfile(self.image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

            # 알파 채널이 있는 경우 처리
            if len(image.shape) > 2 and image.shape[2] == 4:
                # 알파 채널 분리
                alpha = image[:, :, 3]
                # BGR 채널만 선택
                image = image[:, :, :3]
                # 알파 채널을 마스크로 사용하여 흰색 배경과 블렌딩
                white_background = np.ones_like(image) * 255
                alpha = alpha[:, :, np.newaxis] / 255.0
                image = (image * alpha + white_background * (1 - alpha)).astype(np.uint8)

            self.background_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.background_image = cv2.resize(self.background_image, self.target_size)
            self.update_image()

    def update_image(self):
        if self.background_image is None:
            return

        height, width, channel = self.background_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(self.background_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)

    def open_shape_drawing(self):
        if self.background_image is None:
            return

        dialog = ShapeDrawingDialog(self, self.background_image)
        dialog.exec_()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VideoEffectApp()
    ex.show()
    sys.exit(app.exec_())