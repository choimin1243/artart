import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QLabel
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, QTimer, QPoint, QRect


class VideoEffectApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.image_path = None
        self.ellipse_set = False  # 마우스로 영역을 설정했는지 여부

        self.drawing = False
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.ellipse_rect = QRect()

        self.show_red_circle = True  # 빨간 원을 표시할지 여부

    def initUI(self):
        self.setWindowTitle('Video Effect Application')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        btn_layout = QHBoxLayout()
        self.btn_select_image = QPushButton('Select Image', self)
        self.btn_select_image.clicked.connect(self.select_image)
        btn_layout.addWidget(self.btn_select_image)

        self.btn_set_circle = QPushButton('Set Circle', self)
        self.btn_set_circle.clicked.connect(self.toggle_circle_setting)
        btn_layout.addWidget(self.btn_set_circle)

        self.btn_start_video = QPushButton('Start Video', self)
        self.btn_start_video.clicked.connect(self.toggle_video)
        btn_layout.addWidget(self.btn_start_video)

        layout.addLayout(btn_layout)

        self.image_label = QLabel(self)
        self.image_label.setMouseTracking(True)
        self.image_label.setFixedSize(640, 480)  # 기본 크기 설정
        layout.addWidget(self.image_label)

        self.setLayout(layout)

    def select_image(self):
        file_dialog = QFileDialog()
        self.image_path, _ = file_dialog.getOpenFileName(self, 'Select Image', '', 'Image Files (*.png *.jpg *.bmp)')
        if self.image_path:
            pixmap = QPixmap(self.image_path)
            self.image_label.setPixmap(pixmap.scaled(640, 480, Qt.KeepAspectRatio))

    def toggle_circle_setting(self):
        if self.btn_set_circle.text() == 'Set Circle':
            self.btn_set_circle.setText('Finish Setting')
            self.image_label.mousePressEvent = self.mousePressEvent
            self.image_label.mouseMoveEvent = self.mouseMoveEvent
            self.image_label.mouseReleaseEvent = self.mouseReleaseEvent
        else:
            self.btn_set_circle.setText('Set Circle')
            self.image_label.mousePressEvent = None
            self.image_label.mouseMoveEvent = None
            self.image_label.mouseReleaseEvent = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_point = event.pos()
            self.end_point = event.pos()

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            self.end_point = event.pos()
            self.calculate_ellipse()
            self.ellipse_set = True  # 새로운 영역 설정
            self.update()

    def calculate_ellipse(self):
        self.ellipse_rect = QRect(self.start_point, self.end_point).normalized()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.image_label.pixmap() is not None:
            pixmap_copy = self.image_label.pixmap().copy()
            painter = QPainter(pixmap_copy)

            if self.show_red_circle and self.ellipse_set:
                painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
                painter.drawEllipse(self.ellipse_rect)

            self.image_label.setPixmap(pixmap_copy)
            painter.end()

    def toggle_video(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_start_video.setText('Start Video')
            if self.cap:
                self.cap.release()
                self.cap = None
            self.show_red_circle = True  # 비디오 중지 시 빨간 원 다시 표시
        else:
            self.btn_start_video.setText('Stop Video')
            self.show_red_circle = False  # 비디오 시작 시 빨간 원 숨기기

            if not self.cap:
                self.cap = cv2.VideoCapture(0)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.timer.start(30)  # Update every 30 ms
        self.update()  # 화면 갱신

    def update_frame(self):
        if not self.ellipse_set or not self.cap:  # 영역이 설정되지 않거나 카메라가 없으면 업데이트하지 않음
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)  # 좌우 반전

        # 타원 영역에 맞는 마스크 생성
        mask = np.zeros((480, 640), dtype=np.uint8)
        center = ((self.ellipse_rect.left() + self.ellipse_rect.right()) // 2,
                  (self.ellipse_rect.top() + self.ellipse_rect.bottom()) // 2)
        axes = (self.ellipse_rect.width() // 2, self.ellipse_rect.height() // 2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

        # 비디오를 타원 안에 표시
        frame_resized = cv2.resize(frame, (640, 480))
        frame_masked = cv2.bitwise_and(frame_resized, frame_resized, mask=mask)

        # 오일 페인팅 필터 적용
        frame_oil_painting = cv2.xphoto.oilPainting(frame_masked, 7, 1)

        # 알파 채널 추가
        b, g, r = cv2.split(frame_oil_painting)
        alpha = mask
        frame_oil_painting = cv2.merge((b, g, r, alpha))

        # QImage로 변환 (4채널로)
        qt_image = QImage(frame_oil_painting.data, frame_oil_painting.shape[1], frame_oil_painting.shape[0],
                          QImage.Format_RGBA8888)

        # 원래 이미지에 영상 덧씌우기
        if self.image_label.pixmap():
            original_pixmap = self.image_label.pixmap().copy()
            painter = QPainter(original_pixmap)
            painter.drawImage(0, 0, qt_image)
            painter.end()
            self.image_label.setPixmap(original_pixmap)
        else:
            self.image_label.setPixmap(QPixmap.fromImage(qt_image))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VideoEffectApp()
    ex.show()
    sys.exit(app.exec_())