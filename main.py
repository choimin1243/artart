import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QInputDialog, \
    QLabel, QMessageBox, QLineEdit, QFormLayout, QDialogButtonBox, QDialog
from PIL import Image, ImageDraw, ImageFont
import os
import urllib.parse
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, insert, select
from sqlalchemy import update
import requests
from PyQt5.QtGui import QImage, QPixmap, QFontDatabase, QDesktopServices
from PyQt5.QtCore import Qt
from sqlalchemy.orm import Session
from PyQt5.QtGui import QFontDatabase, QImage, QPixmap, QFont
import shutil
import openai
from PyQt5.QtCore import QUrl
import psycopg2
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import math
import importlib.util
from PyQt5.QtWidgets import QSizePolicy

import cv2 as cv
import numpy as np
import mediapipe as mp
import subprocess
from utils import CvFpsCalc
import time


class StartWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        GPT_API = QPushButton('GPT API 새로 넣기', self)
        Edit_button = QPushButton('GPT API 값 수정하기', self)
        AI_button = QPushButton('공익광고 AI포스터 만들기', self)
        start_button = QPushButton('공익광고 홍보영상 만들기', self)
        picto_gram=QPushButton("픽토그램영상만들기",self)
        start_button.clicked.connect(self.open_main_window)
        Edit_button.clicked.connect(self.Edit)
        AI_button.clicked.connect(self.AIPICTURE)
        GPT_API.clicked.connect(self.GPT)
        picto_gram.clicked.connect(self.picto)
        layout.addWidget(GPT_API)
        layout.addWidget(Edit_button)
        layout.addWidget(start_button)
        layout.addWidget(AI_button)
        layout.addWidget(picto_gram)
        self.setLayout(layout)
        self.setWindowTitle('시작하기')
        self.resize(300, 200)
        self.show()
        self.apply_stylesheet()

    def apply_stylesheet(self):
        font_path = 'unz.ttf'
        font_id = QFontDatabase.addApplicationFont(font_path)
        if font_id == -1:
            print("폰트를 로드하는 데 실패했습니다.")
            return
        font_families = QFontDatabase.applicationFontFamilies(font_id)
        if not font_families:
            print("폰트 패밀리 이름을 가져오는 데 실패했습니다.")
            return
        font_family = font_families[0]
        qss = f"""
        QWidget {{
            background-color: #E0FFFF;
            color: #333333;
            font-family: '{font_family}', Arial, sans-serif;
        }}
        QPushButton {{
            background-color: #1ebdbb;
            border: 2px solid #87CEEB;
            color: #FFFFFF;
            padding: 5px;
            font-size: 35px;
            border-radius: 10px;
        }}
        QPushButton:hover {{
            background-color: #0000CD;
        }}
        QPushButton:pressed {{
            background-color: #FF8C00;
        }}
        QTextEdit {{
            background-color: #FFFFFF;
            color: #333333;
            border: 2px solid #FFD700;
            padding: 12px;
            font-size: 18px;
            border-radius: 10px;
        }}
        """
        self.setStyleSheet(qss)

    def Edit(self):
        self.main_window_edit = MainEdit()
        self.main_window_edit.show()

    def GPT(self):
        self.main_window = Mainstart()
        self.main_window.show()

    def AIPICTURE(self):
        self.main_window = AIPICTURE()
        self.main_window.show()

    def open_main_window(self):
        self.main_window = MainWindow()
        self.main_window.resize(800, 600)  # 원하는 크기로 창 크기 설정
        self.main_window.show()

    def picto(self):
        # 새로운 창 생성
        self.picto_window = QWidget()
        self.picto_window.setWindowTitle('픽토그램 영상 만들기')
        self.picto_window.resize(800, 600)

        # 레이아웃 설정
        layout = QVBoxLayout()

        # 버튼 생성
        button1 = QPushButton('배경지정', self.picto_window)
        button2 = QPushButton('픽토그램실행(종료시 ESC클릭)', self.picto_window)


        button1.clicked.connect(self.img2)
        button2.clicked.connect(self.img3)
        layout.addWidget(button1)
        layout.addWidget(button2)

        # 레이아웃을 새로운 창에 설정
        self.picto_window.setLayout(layout)

        # 창에 CSS 스타일 적용
        self.apply_stylesheet_pictogram(self.picto_window)

        # 창 표시
        self.picto_window.show()






    def apply_stylesheet_pictogram(self, widget):
        font_path = 'unz.ttf'
        font_id = QFontDatabase.addApplicationFont(font_path)
        if font_id == -1:
            print("폰트를 로드하는 데 실패했습니다.")
            return
        font_families = QFontDatabase.applicationFontFamilies(font_id)
        if not font_families:
            print("폰트 패밀리 이름을 가져오는 데 실패했습니다.")
            return
        font_family = font_families[0]
        qss = f"""
        QWidget {{
            background-color: #E0FFFF;
            color: #333333;
            font-family: '{font_family}', Arial, sans-serif;
        }}
        QPushButton {{
            background-color: #1ebdbb;
            border: 2px solid #87CEEB;
            color: #FFFFFF;
            padding: 5px;
            font-size: 35px;
            border-radius: 10px;
        }}
        QPushButton:hover {{
            background-color: #0000CD;
        }}
        QPushButton:pressed {{
            background-color: #FF8C00;
        }}
        QTextEdit {{
            background-color: #FFFFFF;
            color: #333333;
            border: 2px solid #FFD700;
            padding: 12px;
            font-size: 18px;
            border-radius: 10px;
        }}
        """
        widget.setStyleSheet(qss)

    def img2(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        # 파일 열기 대화상자 호출
        self.img_path, _ = QFileDialog.getOpenFileName(self, "파일 선택", "",
                                                       "모든 파일 (*);;텍스트 파일 (*.txt);;이미지 파일 (*.png *.jpg);;PDF 파일 (*.pdf)",
                                                       options=options)

        if self.img_path:
            with open(self.img_path, 'rb') as f:
                file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                self.selected_image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

            if self.selected_image is not None:
                QMessageBox.information(self, "성공", "이미지가 성공적으로 로드되었습니다.")
            else:
                QMessageBox.warning(self, "오류", "이미지를 로드할 수 없습니다.")

    def img3(self):
        try:
            print("Executing main() function from 123.py")

            # Get the directory of the current script
            current_dir = os.path.dirname(os.path.abspath(__file__))

            # Construct the full path to 123.py
            script_path = os.path.join(current_dir, '123.py')

            # Check if the file exists
            if not os.path.exists(script_path):
                print(f"Error: {script_path} not found.")
                return

            # Check if image is selected
            if not hasattr(self, 'selected_image') or self.selected_image is None:
                print("Error: No image selected. Please select an image first.")
                QMessageBox.warning(self, "오류", "이미지가 선택되지 않았습니다. 먼저 이미지를 선택해 주세요.")
                return

            # Load the module
            spec = importlib.util.spec_from_file_location("module_123", script_path)
            module_123 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module_123)

            # Check if main function exists and call it
            if hasattr(module_123, 'main') and callable(module_123.main):
                # Execute the main() function with the loaded image
                module_123.main(self.selected_image)
            else:
                print("Error: main() function not found in 123.py")
                QMessageBox.warning(self, "오류", "123.py 파일에서 main() 함수를 찾을 수 없습니다.")

        except Exception as e:
            print(f"An error occurred while executing main() from 123.py: {e}")
            QMessageBox.critical(self, "오류", f"123.py 실행 중 오류가 발생했습니다: {str(e)}")

DATABASE_URL = "postgresql://postgres.kzvxzgridckghifjpfza:choiminseuck@aws-0-ap-northeast-2.pooler.supabase.com:6543/postgres"


class ApiKeyDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("API Key 입력")
        self.resize(300, 100)

        layout = QFormLayout(self)

        self.codenumber = QLineEdit(self)
        self.text = QLineEdit(self)
        layout.addRow("참여코드:", self.codenumber)
        layout.addRow("명화만들기:", self.text)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        layout.addWidget(button_box)

        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

    def accept(self):
        codenumber = self.codenumber.text()
        prompt = self.text.text()
        super().accept()
        engine = create_engine(DATABASE_URL)
        metadata = MetaData()
        example_table = Table('users', metadata, autoload_with=engine)

        with engine.connect() as connection:
            with Session(engine) as session:
                stmt_id = select(example_table).where(example_table.c.user_code == codenumber)
                result_id = session.execute(stmt_id).fetchone()

        print(result_id)
        try:
            if result_id:
                openai.api_key = result_id[3]
                print(result_id[3])
                response = openai.Image.create(
                    model="dall-e-3",
                    prompt=prompt,
                    n=1,
                    size='1024x1024'
                )
                image_url = response['data'][0]['url']
                if image_url:
                    save_path = os.path.join(os.getcwd(), 'generated_image.png')
                    self.download_image(image_url, save_path)
                    time.sleep(2)
                    self.parent.display_image(save_path)
            else:
                QMessageBox.warning(self, "Error", "아이디가 없습니다.")

        except:
            pass
    def download_image(self, url, save_path):
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                f.write(response.content)
            QMessageBox.information(self, "Success", f"Image successfully downloaded: {save_path}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error downloading image: {e}")


class AIPICTURE(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.apply_stylesheet()
        self.rect_start = None
        self.rect_end = None
        self.current_image_path = None
        self.current_image = None
        self.drawing = False
        self.font_path = 'unz.ttf'  # 폰트 경로를 저장
        self.temp_image = None  # 직사각형을 그릴 임시 이미지를 저장

    def initUI(self):
        main_layout = QHBoxLayout(self)

        left_layout = QVBoxLayout()
        button0 = QPushButton("AI 그림생성하기", self)
        buttonupload=QPushButton("그림불러오기",self)
        button1 = QPushButton('글자넣기', self)
        button2 = QPushButton('도형넣기', self)
        button3 = QPushButton('이미지 저장하기', self)
        button4 = QPushButton('전시회 연결하기', self)

        button0.setFixedSize(200, 70)
        buttonupload.setFixedSize(200,70)
        button1.setFixedSize(200, 70)
        button2.setFixedSize(200, 70)
        button3.setFixedSize(200, 70)
        button4.setFixedSize(200, 70)

        right_layout = QVBoxLayout()
        right_layout.addWidget(button0)
        right_layout.addWidget(buttonupload)
        right_layout.addWidget(button1)
        right_layout.addWidget(button2)
        right_layout.addWidget(button3)
        right_layout.addWidget(button4)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap('gongik.png')
        small_pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # QLabel에 줄인 QPixmap 적용
        self.image_label.setPixmap(small_pixmap)
        self.image_label.setScaledContents(True)  # 이미지가 QLabel 안에 맞도록 설정


        main_layout.addLayout(left_layout, 1)
        main_layout.addWidget(self.image_label, 3)
        main_layout.addLayout(right_layout, 1)
        button0.clicked.connect(self.AIDRAW)
        button1.clicked.connect(self.add_draw)
        buttonupload.clicked.connect(self.upload)
        button2.clicked.connect(self.add_text_mode)
        button3.clicked.connect(self.save)
        button4.clicked.connect(self.connection)
        self.setWindowTitle('미술프로그램')
        self.resize(1200, 800)

    def connection(self):
        code, ok = QInputDialog.getText(self, '코드 입력', '코드를 입력하세요:')
        if ok:
            engine = create_engine(DATABASE_URL)
            metadata = MetaData()
            example_table = Table('users', metadata, autoload_with=engine)

            with engine.connect() as connection:
                with Session(engine) as session:
                    stmt = select(example_table).where(example_table.c.user_code == code)
                    result = session.execute(stmt).fetchone()

            if result:
                url = result[4]  # Assuming the URL is stored in the 5th column
                if url:
                    QDesktopServices.openUrl(QUrl(url))
                else:
                    QMessageBox.warning(self, "Error", "URL이 없습니다.")
            else:
                QMessageBox.warning(self, "Error", "해당 코드에 대한 정보가 없습니다.")

    def upload(self):
        try:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            file_name, _ = QFileDialog.getOpenFileName(self, "이미지 파일 선택", "",
                                                       "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)",
                                                       options=options)
            if file_name:
                # 파일 경로를 UTF-8로 인코딩
                file_name_utf8 = file_name.encode('utf-8')

                # numpy array로 파일 읽기
                np_arr = np.fromfile(file_name_utf8, np.uint8)

                # cv2로 이미지 디코딩
                self.current_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

                if self.current_image is not None:
                    # BGR에서 RGB로 변환
                    rgb_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)

                    # numpy 배열을 QImage로 변환
                    height, width, channel = rgb_image.shape
                    bytes_per_line = 3 * width
                    q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

                    # QImage를 QPixmap으로 변환
                    pixmap = QPixmap.fromImage(q_image)

                    # QLabel에 QPixmap 설정
                    self.image_label.setPixmap(pixmap)
                    self.image_label.setScaledContents(True)

                    self.current_image_path = file_name
                    self.temp_image = self.current_image.copy()

                    QMessageBox.information(self, "성공", "이미지가 성공적으로 로드되었습니다.")
                else:
                    QMessageBox.warning(self, "오류", "선택한 파일을 이미지로 불러올 수 없습니다.")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"이미지 로드 중 오류가 발생했습니다: {str(e)}")
    def save(self):
        if self.image_label.pixmap() is not None:
            # 파일 저장 대화상자를 통해 저장할 경로를 선택합니다.
            save_path, _ = QFileDialog.getSaveFileName(self, "이미지 저장", "", "PNG Files (*.png);;All Files (*)")
            if save_path:
                # QLabel의 QPixmap을 QImage로 변환합니다.
                qimage = self.image_label.pixmap().toImage()

                # QImage를 PIL 이미지로 변환합니다.
                buffer = qimage.bits().asstring(qimage.width() * qimage.height() * qimage.depth() // 8)
                pil_image = Image.frombuffer("RGBA", (qimage.width(), qimage.height()), buffer, "raw", "BGRA", 0, 1)

                # PIL 이미지를 저장합니다.
                pil_image.save(save_path)
                QMessageBox.information(self, "저장 완료", "이미지가 성공적으로 저장되었습니다.")
        else:
            QMessageBox.warning(self, "저장 실패", "저장할 이미지가 없습니다.")

    def add_draw(self):
        if self.current_image is None:
            QMessageBox.warning(self, "Error", "먼저 이미지를 로드하십시오.")
            return
        self.image_label.setCursor(Qt.CrossCursor)
        self.image_label.mousePressEvent = self.start_drawing
        self.image_label.mouseMoveEvent = self.update_drawing
        self.image_label.mouseReleaseEvent = self.end_drawings

    def AIDRAW(self):
        self.api_key_dialog = ApiKeyDialog(self)
        self.api_key_dialog.exec_()

    def display_image(self, image_path):
        self.current_image_path = image_path
        self.current_image = cv2.imread(image_path)
        self.temp_image = self.current_image.copy()  # 임시 이미지 초기화
        self.update_label()

    def update_label(self):
        if self.temp_image is not None:
            height, width, channel = self.temp_image.shape
            bytes_per_line = 3 * width
            q_img = QImage(self.temp_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)

    def add_text_mode(self):
        if self.current_image is None:
            QMessageBox.warning(self, "Error", "먼저 이미지를 로드하십시오.")
            return
        self.image_label.setCursor(Qt.CrossCursor)
        self.image_label.mousePressEvent = self.start_drawing
        self.image_label.mouseMoveEvent = self.update_drawing
        self.image_label.mouseReleaseEvent = self.end_drawing

    def start_drawing(self, event):
        self.rect_start = (event.pos().x(), event.pos().y())
        self.drawing = True

    def update_drawing(self, event):
        if self.drawing:
            self.rect_end = (event.pos().x(), event.pos().y())
            self.temp_image = self.current_image.copy()
            cv2.rectangle(self.temp_image, self.rect_start, self.rect_end, (255, 0, 0), 2)  # 파란색 직사각형
            self.update_label()

    def end_drawing(self, event):
        if not self.drawing:
            return
        self.rect_end = (event.pos().x(), event.pos().y())
        self.drawing = False
        self.image_label.setCursor(Qt.ArrowCursor)
        print("hello12")
        self.draw_rectangle_on_image()

    def draw_rectangle_on_image(self):
        if self.current_image is None:
            return

        x1, y1 = self.rect_start
        x2, y2 = self.rect_end

        # Convert OpenCV image (numpy array) to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB))

        # Draw the rectangle on the image
        draw = ImageDraw.Draw(pil_image)

        color, ok = self.get_color()

        if color == 'red':
            draw.rectangle([self.rect_start, self.rect_end], outline='blue', width=2, fill='blue')
        else:
            draw.rectangle([self.rect_start, self.rect_end], outline=color, width=2, fill=color)

        # Convert PIL image back to OpenCV image
        self.current_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        self.temp_image = self.current_image.copy()  # 직사각형이 추가된 이미지를 임시 이미지로 설정
        self.update_label()

        # Save the updated image with the rectangle
        if self.current_image_path:
            cv2.imwrite(self.current_image_path, self.current_image)

    def add_text_to_image(self, text, color):
        print(color)
        if self.current_image is None:
            return

        x1, y1 = self.rect_start
        x2, y2 = self.rect_end
        text_position = (min(x1, x2), min(y1, y2) + (abs(y2 - y1) // 2))

        rect_height = abs(y2 + 20 - y1)
        print(rect_height)

        font_size = max(10, rect_height // 2)  # 최소 폰트 크기를 10으로 설정

        # Convert OpenCV image (numpy array) to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB))

        # Load the font
        font = ImageFont.truetype(self.font_path, font_size)

        # Draw the text on the image
        draw = ImageDraw.Draw(pil_image)

        if color == 'red':
            draw.text(text_position, text, font=font, fill='blue')
        else:
            draw.text(text_position, text, font=font, fill=color)

        # Convert PIL image back to OpenCV image
        self.current_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        self.temp_image = self.current_image.copy()  # 텍스트가 추가된 이미지를 임시 이미지로 설정
        self.update_label()

        # Save the updated image with text
        if self.current_image_path:
            cv2.imwrite(self.current_image_path, self.current_image)

    def end_drawings(self, event):
        if not self.drawing:
            return
        self.rect_end = (event.pos().x(), event.pos().y())
        self.drawing = False
        self.image_label.setCursor(Qt.ArrowCursor)
        self.prompt_text_input()

    def prompt_text_input(self):
        if self.rect_start and self.rect_end:
            text, ok = QInputDialog.getText(self, '텍스트 입력', '사각형 내부에 넣을 텍스트를 입력하세요:')
            if ok and text:
                color, ok = self.get_color()
                if ok:
                    self.add_text_to_image(text, color)

    def get_color(self):
        colors = ['white', 'black', 'green', 'red']
        color, ok = QInputDialog.getItem(self, "색깔 선택", "텍스트 색깔을 선택하세요:", colors, 0, False)
        return color, ok

    def apply_stylesheet(self):
        font_path = 'unz.ttf'
        font_id = QFontDatabase.addApplicationFont(font_path)
        if font_id == -1:
            print("폰트를 로드하는 데 실패했습니다.")
            return
        font_families = QFontDatabase.applicationFontFamilies(font_id)
        if not font_families:
            print("폰트 패밀리 이름을 가져오는 데 실패했습니다.")
            return
        font_family = font_families[0]
        qss = f"""
        QWidget {{
            background-color: #E0FFFF;
            color: #333333;
            font-family: '{font_family}', Arial, sans-serif;
        }}
        QPushButton {{
            background-color: #1ebdbb;
            border: 2px solid #87CEEB;
            color: #FFFFFF;
            padding: 5px;
            font-size: 35px;
            border-radius: 10px;
        }}
        QPushButton:hover {{
            background-color: #0000CD;
        }}
        QPushButton:pressed {{
            background-color: #FF8C00;
        }}
        QTextEdit {{
            background-color: #FFFFFF;
            color: #333333;
            border: 2px solid #FFD700;
            padding: 12px;
            font-size: 18px;
            border-radius: 10px;
        }}
        """
        self.setStyleSheet(qss)


class MainEdit(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.apply_stylesheet()

    def apply_stylesheet(self):
        font_path = 'unz.ttf'
        font_id = QFontDatabase.addApplicationFont(font_path)
        if font_id == -1:
            print("폰트를 로드하는 데 실패했습니다.")
            return
        font_families = QFontDatabase.applicationFontFamilies(font_id)
        if not font_families:
            print("폰트 패밀리 이름을 가져오는 데 실패했습니다.")
            return
        font_family = font_families[0]
        qss = f"""
        QWidget {{
            background-color: #E0FFFF;
            color: #333333;
            font-family: '{font_family}', Arial, sans-serif;
        }}
        QPushButton {{
            background-color: #1ebdbb;
            border: 2px solid #87CEEB;
            color: #FFFFFF;
            padding: 5px;
            font-size: 35px;
            border-radius: 10px;
        }}
        QPushButton:hover {{
            background-color: #0000CD;
        }}
        QPushButton:pressed {{
            background-color: #FF8C00;
        }}
        QTextEdit {{
            background-color: #FFFFFF;
            color: #333333;
            border: 2px solid #FFD700;
            padding: 12px;
            font-size: 18px;
            border-radius: 10px;
        }}
        """
        self.setStyleSheet(qss)

    def initUI(self):
        main_layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        font = QFont()
        font.setPointSize(16)

        self.id_label = QLabel('아이디:')
        self.id_label.setFont(font)
        self.id_input = QLineEdit(self)
        self.id_input.setFont(font)
        form_layout.addRow(self.id_label, self.id_input)

        self.code_label = QLabel('참여코드:')
        self.code_label.setFont(font)
        self.code_input = QLineEdit(self)
        self.code_input.setFont(font)
        form_layout.addRow(self.code_label, self.code_input)

        self.api_label = QLabel('수정된API 키:')
        self.api_label.setFont(font)
        self.api_input = QLineEdit(self)
        self.api_input.setFont(font)
        form_layout.addRow(self.api_label, self.api_input)

        self.url_label = QLabel('수정된 url:')
        self.url_label.setFont(font)
        self.url_input = QLineEdit(self)
        self.url_input.setFont(font)
        form_layout.addRow(self.url_label, self.url_input)

        self.submit_button2 = QPushButton('제출', self)
        self.submit_button2.setFont(font)
        self.submit_button2.clicked.connect(self.submited2)
        form_layout.addRow(self.submit_button2)

        main_layout.addLayout(form_layout)

        self.setWindowTitle('미술프로그램')
        self.resize(200, 200)
        self.show()

    def submited2(self):
        DATABASE_URL = "postgresql://postgres.kzvxzgridckghifjpfza:choiminseuck@aws-0-ap-northeast-2.pooler.supabase.com:6543/postgres"

        engine = create_engine(DATABASE_URL)
        metadata = MetaData()
        example_table = Table('users', metadata, autoload_with=engine)
        print("hello")

        user_id = self.id_input.text()
        user_code = self.code_input.text()
        user_api_key = self.api_input.text()
        url_input = self.url_input.text()
        print(user_id)
        with engine.connect() as connection:
            with Session(engine) as session:
                stmt_id = select(example_table).where(
                    example_table.c.user_name == user_id,
                    example_table.c.user_code == user_code
                )
                result_id = session.execute(stmt_id).fetchone()
                print(result_id)

                if result_id:
                    stmt_update = update(example_table).where(
                        example_table.c.user_name == user_id,
                        example_table.c.user_code == user_code
                    ).values(user_api_key=user_api_key, url_input=url_input)

                    session.execute(stmt_update)
                    session.commit()
                    print(f"API_KEY for user_id {user_id} updated successfully.")
                    QMessageBox.information(self, "수정 성공", "수정이 성공되었습니다.", QMessageBox.Ok)
                else:
                    print("No matching user_id and user_code found.")
                    QMessageBox.warning(self, "수정 실패", "일치하는 사용자 ID와 참여코드를 찾을 수 없습니다.", QMessageBox.Ok)


class Mainstart(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.apply_stylesheet()

    def apply_stylesheet(self):
        font_path = 'unz.ttf'
        font_id = QFontDatabase.addApplicationFont(font_path)
        if font_id == -1:
            print("폰트를 로드하는 데 실패했습니다.")
            return
        font_families = QFontDatabase.applicationFontFamilies(font_id)
        if not font_families:
            print("폰트 패밀리 이름을 가져오는 데 실패했습니다.")
            return
        font_family = font_families[0]
        qss = f"""
        QWidget {{
            background-color: #E0FFFF;
            color: #333333;
            font-family: '{font_family}', Arial, sans-serif;
        }}
        QPushButton {{
            background-color: #1ebdbb;
            border: 2px solid #87CEEB;
            color: #FFFFFF;
            padding: 5px;
            font-size: 35px;
            border-radius: 10px;
        }}
        QPushButton:hover {{
            background-color: #0000CD;
        }}
        QPushButton:pressed {{
            background-color: #FF8C00;
        }}
        QTextEdit {{
            background-color: #FFFFFF;
            color: #333333;
            border: 2px solid #FFD700;
            padding: 12px;
            font-size: 18px;
            border-radius: 10px;
        }}
        """
        self.setStyleSheet(qss)

    def initUI(self):
        main_layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        font = QFont()
        font.setPointSize(16)

        self.id_label = QLabel('아이디:')
        self.id_label.setFont(font)
        self.id_input = QLineEdit(self)
        self.id_input.setFont(font)
        form_layout.addRow(self.id_label, self.id_input)

        self.code_label = QLabel('참여코드:')
        self.code_label.setFont(font)
        self.code_input = QLineEdit(self)
        self.code_input.setFont(font)
        form_layout.addRow(self.code_label, self.code_input)

        self.api_label = QLabel('API 키:')
        self.api_label.setFont(font)
        self.api_input = QLineEdit(self)
        self.api_input.setFont(font)
        form_layout.addRow(self.api_label, self.api_input)

        self.url = QLabel('전시회 사이트주소:')
        self.url.setFont(font)
        self.url_input = QLineEdit(self)
        self.url_input.setFont(font)
        form_layout.addRow(self.url, self.url_input)

        self.submit_button = QPushButton('제출', self)
        self.submit_button.setFont(font)
        self.submit_button.clicked.connect(self.submit)
        form_layout.addRow(self.submit_button)

        main_layout.addLayout(form_layout)

        self.setWindowTitle('미술프로그램')
        self.resize(400, 200)
        self.show()

    def submit(self):
        db_url = "postgresql://postgres.kzvxzgridckghifjpfza:choiminseuck@aws-0-ap-northeast-2.pooler.supabase.com:6543/postgres"

        try:
            conn = psycopg2.connect(db_url)
            cur = conn.cursor()

            # 테이블이 없으면 생성하는 SQL
            create_table_query = """
                CREATE TABLE IF NOT EXISTS users (
                    user_id SERIAL PRIMARY KEY,
                    user_name VARCHAR(100),
                    user_code VARCHAR(50),
                    user_api_key VARCHAR(100),
                    url_input TEXT
                );
                """

            # 테이블 생성 쿼리 실행
            cur.execute(create_table_query)
            conn.commit()

            print("테이블이 성공적으로 생성되었거나 이미 존재합니다.")

            # 삽입할 데이터
            user_name = self.id_input.text()
            user_code = self.code_input.text()
            user_api_key = self.api_input.text()
            url_input = self.url_input.text()

            select_name_query = """
                SELECT user_id FROM users WHERE user_name = %s;
            """
            cur.execute(select_name_query, (user_name,))
            name_exists = cur.fetchone()

            # user_code 존재 여부 확인 쿼리
            select_code_query = """
                SELECT user_id FROM users WHERE user_code = %s;
            """
            cur.execute(select_code_query, (user_code,))
            code_exists = cur.fetchone()

            if name_exists:
                QMessageBox.warning(self, "오류", "해당 사용자 이름이 이미 존재합니다.")
            elif code_exists:
                QMessageBox.warning(self, "오류", "해당 사용자 코드는 이미 존재합니다.")
            else:
                # 데이터 삽입 쿼리
                insert_query = """
                    INSERT INTO users (user_name, user_code, user_api_key, url_input)
                    VALUES (%s, %s, %s, %s);
                """
                cur.execute(insert_query, (user_name, user_code, user_api_key, url_input))
                conn.commit()  # 삽입 후 변경 사항 저장
                QMessageBox.information(self, "성공", "아이디가 성공적으로 생성되었습니다!")

        except Exception as e:
            QMessageBox.critical(self, "오류", f"오류 발생: {e}")

        finally:
            # 커서와 연결 종료
            if cur:
                cur.close()
            if conn:
                conn.close()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.flip_states = []
        self.apply_stylesheet()
        self.mouse_position = None
        self.text_info = None
        self.background = None
        self.images = []
        self.all_objects = []
        self.rotations = []
        self.drawing = False
        self.dragging_object_index = -1
        self.drag_offset = (0, 0)
        self.is_running = False
        self.rect_start = None
        self.rect_end = None
        self.clone = None

    def initUI(self):
        main_layout = QHBoxLayout(self)

        left_layout = QVBoxLayout()
        button0 = QPushButton("색지우기", self)
        button1 = QPushButton('물체넣기', self)
        button2 = QPushButton('배경넣기', self)

        button0.setFixedSize(150, 100)
        button1.setFixedSize(150, 100)
        button2.setFixedSize(150, 100)

        right_layout = QVBoxLayout()
        left_layout.setSpacing(10)  # 버튼 간의 간격을 10으로 설정

        button5 = QPushButton('글귀넣기', self)
        button3 = QPushButton('영상 실행', self)
        button4 = QPushButton('영상종료', self)
        button6 = QPushButton('입체영상', self)
        button8 = QPushButton('이미지 다듬기', self)

        button5.setFixedSize(150, 100)
        button3.setFixedSize(150, 100)
        button4.setFixedSize(150, 100)
        button6.setFixedSize(150, 100)

        right_layout.addWidget(button0)
        right_layout.addWidget(button8)
        right_layout.addWidget(button1)
        right_layout.addWidget(button2)
        right_layout.addWidget(button5)
        right_layout.addWidget(button3)
        right_layout.addWidget(button6)
        right_layout.addWidget(button4)
        right_layout.setSpacing(10)  # 버튼 간의 간격을 10으로 설정

        self.image_label = QLabel(self)
        pixmap = QPixmap('gongik.png')
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)

        main_layout.addLayout(left_layout, 1)
        main_layout.addWidget(self.image_label, 3)
        main_layout.addLayout(right_layout, 1)

        button0.clicked.connect(self.make)
        button1.clicked.connect(self.object)
        button2.clicked.connect(self.open_new_window2)

        button5.clicked.connect(self.show_image_with_rectangle_selection)
        button3.clicked.connect(self.open_new_window3)
        button4.clicked.connect(self.open_new_window4)
        button6.clicked.connect(self.open_new_window6)
        button8.clicked.connect(self.erase_image)
        left_layout.setContentsMargins(0, 0, 0, 0)  # 외부 여백 없애기
        right_layout.setContentsMargins(0, 0, 0, 0)  # 외부 여백 없애기

        small_pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # QLabel에 줄인 QPixmap 적용
        self.image_label.setPixmap(small_pixmap)
        self.image_label.setScaledContents(True)  # 이미지가 QLabel 안에 맞도록 설정

        # 메인 레이아웃에 왼쪽, 이미지, 오른쪽 레이아웃 추가
        main_layout.addLayout(left_layout, 1)
        main_layout.addWidget(self.image_label, 3)
        main_layout.addLayout(right_layout, 1)

        # 창의 크기 조절을 가능하게 설정
        self.setWindowTitle('공익광고 홍보영상 프로그램')
        self.resize(600, 400)  # 기본 크기 설정
        self.setMinimumSize(500, 300)  # 최소 크기 설정

        self.setWindowTitle('미술프로그램')
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def apply_stylesheet(self):
        font_path = 'unz.ttf'
        font_id = QFontDatabase.addApplicationFont(font_path)
        if font_id == -1:
            print("폰트를 로드하는 데 실패했습니다.")
            return
        font_families = QFontDatabase.applicationFontFamilies(font_id)
        if not font_families:
            print("폰트 패밀리 이름을 가져오는 데 실패했습니다.")
            return
        font_family = font_families[0]
        qss = f"""
        QWidget {{
            background-color: #E0FFFF;
            color: #333333;
            font-family: '{font_family}', Arial, sans-serif;
        }}
        QPushButton {{
            background-color: #1ebdbb;
            border: 2px solid #87CEEB;
            color: #FFFFFF;
            padding: 5px;
            font-size: 35px;
            border-radius: 10px;
        }}
        QPushButton:hover {{
            background-color: #0000CD;
        }}
        QPushButton:pressed {{
            background-color: #FF8C00;
        }}
        QTextEdit {{
            background-color: #FFFFFF;
            color: #333333;
            border: 2px solid #FFD700;
            padding: 12px;
            font-size: 18px;
            border-radius: 10px;
        }}
        """
        self.setStyleSheet(qss)

    def onMouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            cv2.circle(self.img_edit, (x, y), 30, (255, 255, 255), -1)
            cv2.imshow("수정", self.img_edit)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                cv2.circle(self.img_edit, (x, y), 30, (255, 255, 255), -1)
                cv2.imshow("수정", self.img_edit)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

    def erase_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "이미지 파일 선택", "",
                                                   "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)", options=options)
        if file_name:
            try:
                with open(file_name, 'rb') as f:
                    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                    self.img_edit = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
                    self.img_original = self.img_edit.copy()
                    cv2.namedWindow("수정")
                    cv2.setMouseCallback("수정", self.onMouse)

                    while True:
                        cv2.imshow("수정", self.img_edit)
                        k = cv2.waitKey(1) & 0xFF
                        if k == 27:  # ESC key
                            break

                    cv2.destroyAllWindows()

                    # 수정된 이미지 저장
                    file_name_without_ext = os.path.splitext(file_name)[0]
                    new_file_name = f"{file_name_without_ext}_erased.png"

                    # 절대 경로 사용
                    abs_path = os.path.abspath(new_file_name)
                    print(f"Trying to save file to: {abs_path}")  # 디버깅용 출력

                    # cv2.imwrite 대신 PIL 사용
                    from PIL import Image
                    img_pil = Image.fromarray(cv2.cvtColor(self.img_edit, cv2.COLOR_BGR2RGB))
                    img_pil.save(abs_path)

                    if os.path.exists(abs_path):
                        print(f"File successfully saved to: {abs_path}")  # 디버깅용 출력
                        QMessageBox.information(self, "저장 완료", f"수정된 이미지가 {abs_path}에 저장되었습니다.")
                    else:
                        print(f"File not found at: {abs_path}")  # 디버깅용 출력
                        QMessageBox.warning(self, "저장 실패", "파일 저장에 실패했습니다.")

            except Exception as e:
                print(f"Error: {e}")  # 디버깅용 출력
                QMessageBox.warning(self, "오류", f"이미지 처리 중 오류가 발생했습니다: {e}")
    def onMouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            cv2.circle(self.img_edit, (x, y), 30, (255, 255, 255), -1)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                cv2.circle(self.img_edit, (x, y), 30, (255, 255, 255), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

    def open_new_window6(self):
        try:
            self.final_running = True
            if hasattr(sys, '_MEIPASS'):
                base_path = sys._MEIPASS
            else:
                base_path = os.path.dirname(os.path.abspath(__file__))

            fname = QFileDialog.getOpenFileName(self, "Open File", "./")
            if fname[0]:
                print(fname[0])
                file_path = urllib.parse.quote(fname[0])
                file_path = urllib.parse.unquote(file_path)
                video1 = cv2.VideoCapture(file_path)
                video2 = cv2.VideoCapture(file_path)
                video3 = cv2.VideoCapture(file_path)
                video4 = cv2.VideoCapture(file_path)

                if not video1.isOpened() or not video2.isOpened() or not video3.isOpened() or not video4.isOpened():
                    print("Error: One or more video files could not be opened.")
                    return

                self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                self.out = cv2.VideoWriter('result_final.avi', self.fourcc, 20.0, (3000, 2400))

                if not self.out.isOpened():
                    print("Error: Could not open output file.")
                    return

                while self.final_running:
                    ret1, frame1 = video1.read()
                    ret2, frame2 = video2.read()
                    ret3, frame3 = video3.read()
                    ret4, frame4 = video4.read()

                    if not ret1 or not ret2 or not ret3 or not ret4:
                        break

                    frame1 = cv2.resize(frame1, (800, 800))
                    frame1 = frame1[500:, 400:600, :]
                    frame1 = cv2.resize(frame1, (800, 800))

                    frame4 = cv2.resize(frame4, (800, 800))
                    middle = frame4[120:500, 450:600, :]
                    frame4 = cv2.resize(middle, (800, 800))

                    frame2 = cv2.resize(frame2, (800, 800))
                    frame2 = frame2[0:500, :450, :]
                    frame2 = cv2.resize(frame2, (800, 800))

                    frame3 = cv2.resize(frame3, (800, 800))
                    right = frame3[10:510, 620:, :]
                    frame3 = cv2.resize(right, (800, 800))

                    rows, cols, _ = frame1.shape

                    pts1 = np.float32([[0, 0], [0, rows], [cols, 0], [cols, rows]])
                    pts2 = np.float32([[300, 250], [10, rows - 50], [cols - 300, 250], [cols - 10, rows - 50]])
                    pts3 = np.float32([[0, 0], [0, rows], [cols, 0], [cols, rows]])
                    pts4 = np.float32([[546, 9], [500, 250], [780, 11], [788, 750]])

                    pts5 = np.float32([[0, 0], [0, rows], [cols, 0], [cols, rows]])
                    pts6 = np.float32([[6, 12], [14, 742], [256, 4], [302, 248]])

                    pts7 = np.float32([[0, 0], [0, rows], [cols, 0], [cols, rows]])
                    pts8 = np.float32([[251, 4], [306, 250], [550, 7], [505, 247]])

                    mtrx1 = cv2.getPerspectiveTransform(pts1, pts2)
                    dst1 = cv2.warpPerspective(frame1, mtrx1, (cols, rows))

                    mtrx2 = cv2.getPerspectiveTransform(pts3, pts4)
                    dst2 = cv2.warpPerspective(frame3, mtrx2, (cols, rows))

                    dst2 = cv2.resize(dst2, (cols, rows))

                    combined_frame = cv2.add(dst1, dst2)

                    mtrx3 = cv2.getPerspectiveTransform(pts5, pts6)
                    dst3 = cv2.warpPerspective(frame2, mtrx3, (cols, rows))

                    combined_frame = cv2.add(combined_frame, dst3)

                    mtrx4 = cv2.getPerspectiveTransform(pts7, pts8)
                    dst4 = cv2.warpPerspective(frame4, mtrx4, (cols, rows))

                    combined_frame = cv2.add(combined_frame, dst4)

                    combined_frame = cv2.resize(combined_frame, (3000, 2400))

                    cv2.imshow("합성된 영상", combined_frame)

                    self.out.write(combined_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                video1.release()
                video2.release()
                video3.release()
                video4.release()
                self.out.release()
                cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error: {e}")

    def make(self):
        try:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            files, _ = QFileDialog.getOpenFileNames(self, "파일 선택", "", "모든 파일 (*);;텍스트 파일 (*.txt)", options=options)

            image_list = files
            self.images_background = []

            for img_path in image_list:
                try:
                    with open(img_path, 'rb') as f:
                        file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
                        self.images_background.append(img)
                except Exception as e:
                    print(f"Exception loading image {img_path}: {e}")

            for i in range(len(self.images_background)):
                gray = cv2.cvtColor(self.images_background[i], cv2.COLOR_BGR2GRAY)

                blur = cv2.GaussianBlur(gray, (5, 5), 0)

                edges = cv2.Canny(blur, 50, 150)

                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                new_img = np.zeros_like(self.images_background[i])

                cv2.drawContours(new_img, contours, -1, (255, 255, 255), thickness=-1)

                new_img = cv2.bitwise_not(new_img)

                cv2.imshow("result", new_img)

                filename = f"result_image_{i}.png"
                cv2.imwrite(filename, new_img)

        except Exception as e:
            print(f"Error in make: {e}")

    def myPutText(self, src, text, pos, font_size, font_color):
        try:
            img_pil = Image.fromarray(src)
            draw = ImageDraw.Draw(img_pil)
            font = ImageFont.truetype('unz.ttf', 60)
            draw.text(pos, text, font=font, fill=font_color)
            return np.array(img_pil)

        except Exception as e:
            print(f"Error in myPutText: {e}")
            return src

    def show_image_with_rectangle_selection(self):
        try:
            if self.background is None:
                print("No background image loaded.")
                return

            self.clone = self.background.copy()
            cv2.imshow("image", self.clone)
            cv2.setMouseCallback("image", self.extract_coordinates)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            if self.rect_start and self.rect_end:
                text, ok = QInputDialog.getText(self, '텍스트 입력', '사각형 내부에 넣을 텍스트를 입력하세요:')
                if ok and text:
                    self.text_info = {'text': text, 'rect_start': self.rect_start, 'rect_end': self.rect_end}
                    self.add_text_to_background()

        except Exception as e:
            print(f"Error in show_image_with_rectangle_selection: {e}")

    def add_text_to_frame(self, frame, text_info):
        try:
            img_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pil)
            font = ImageFont.truetype('unz.ttf', text_info['font_size'])
            draw.text(text_info['position'], text_info['text'], font=font, fill=text_info['font_color'])
            return np.array(img_pil)

        except Exception as e:
            print(f"Error in add_text_to_frame: {e}")
            return frame

    def add_text_to_background(self):
        try:
            if self.background is None:
                print("No background image loaded.")
                return

            img_pil = Image.fromarray(cv2.cvtColor(self.background, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            font = ImageFont.truetype('unz.ttf', 60)

            text = self.text_info['text']
            rect_start = self.text_info['rect_start']
            rect_end = self.text_info['rect_end']
            x = rect_start[0]
            y = rect_start[1]
            width = rect_end[0] - rect_start[0]
            height = rect_end[1] - rect_start[1]
            font = ImageFont.truetype('unz.ttf', height)

            draw.text((x - 100 + width // 2, y - 50 + height // 2), text, font=font, fill=(255, 255, 255, 255))

            img_with_text = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            self.background = img_with_text
            self.update_image_label(img_with_text)

        except Exception as e:
            print(f"Error in add_text_to_background: {e}")

    def extract_coordinates(self, event, x, y, flags, param):
        try:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.rect_start = (x, y)
                self.rect_end = None
                self.clone = self.background.copy()

            elif event == cv2.EVENT_MOUSEMOVE:
                if self.drawing:
                    self.rect_end = (x, y)
                    image = self.clone.copy()
                    cv2.rectangle(image, self.rect_start, self.rect_end, (0, 255, 0), 2)
                    cv2.imshow("image", image)

            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                self.rect_end = (x, y)
                cv2.rectangle(self.clone, self.rect_start, self.rect_end, (0, 255, 0), 2)
                cv2.imshow("image", self.clone)

        except Exception as e:
            print(f"Error in extract_coordinates: {e}")

    def object(self):
        try:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            files, _ = QFileDialog.getOpenFileNames(self, "파일 선택", "",
                                                    "이미지 파일 (*.png *.jpg);;모든 파일 (*)", options=options)

            for img_path in files:
                with open(img_path, 'rb') as f:
                    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

                    # 알파 채널 처리
                    if img is not None:
                        if len(img.shape) == 2:  # 그레이스케일 이미지
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
                        elif len(img.shape) == 3:
                            if img.shape[2] == 3:  # BGR 이미지
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                            elif img.shape[2] == 4:  # BGRA 이미지
                                pass

                        # 알파 채널이 있는 경우 처리
                        if img.shape[2] == 4:
                            alpha = img[:, :, 3]
                            white_background = np.ones_like(img[:, :, :3]) * 255
                            alpha_3d = alpha[:, :, np.newaxis] / 255.0
                            img_rgb = img[:, :, :3]
                            img_new = (img_rgb * alpha_3d + white_background * (1 - alpha_3d)).astype(np.uint8)
                        else:
                            img_new = img[:, :, :3]

                        position = (np.random.randint(0, self.background.shape[1] - 100),
                                    np.random.randint(0, self.background.shape[0] - 100))
                        self.all_objects.append((img_new, position))
                        self.rotations.append(0)

            self.flip_states = [False] * len(self.all_objects)
            self.rotations = [0] * len(self.rotations)
            self.display_objects_on_background()

        except Exception as e:
            print(f"Error in object: {e}")

    def open_new_window2(self):
        try:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            self.img_path, _ = QFileDialog.getOpenFileName(
                self,
                "배경 이미지 선택",
                "",
                "이미지 파일 (*.png *.jpg *.jpeg *.bmp);;모든 파일 (*)",
                options=options
            )

            if self.img_path:
                try:
                    # 이미지 로드
                    with open(self.img_path, 'rb') as f:
                        file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

                    if img is None:
                        QMessageBox.warning(self, "오류", "이미지를 로드할 수 없습니다.")
                        return

                    # 이미지 채널 처리
                    if len(img.shape) == 2:  # 그레이스케일
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    elif len(img.shape) == 4:  # RGBA
                        # 알파 채널이 있는 경우 처리
                        alpha = img[:, :, 3]
                        img_rgb = img[:, :, :3]
                        white_background = np.ones_like(img_rgb) * 255
                        alpha_3d = alpha[:, :, np.newaxis] / 255.0
                        img = (img_rgb * alpha_3d + white_background * (1 - alpha_3d)).astype(np.uint8)

                    # BGR에서 RGB로 변환
                    self.background = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # 이미지 크기 정규화 (선택적)
                    target_width = 800  # 원하는 너비
                    aspect_ratio = self.background.shape[1] / self.background.shape[0]
                    target_height = int(target_width / aspect_ratio)
                    self.background = cv2.resize(self.background, (target_width, target_height),
                                                 interpolation=cv2.INTER_LANCZOS4)

                    # QImage로 변환
                    height, width, channel = self.background.shape
                    bytes_per_line = 3 * width
                    q_img = QImage(self.background.data, width, height, bytes_per_line, QImage.Format_RGB888)

                    # QPixmap으로 변환 및 화면에 표시
                    pixmap = QPixmap.fromImage(q_img)
                    self.image_label.setPixmap(pixmap)
                    self.image_label.setScaledContents(True)

                    # GUI 업데이트
                    QApplication.processEvents()

                except Exception as e:
                    QMessageBox.warning(self, "오류", f"이미지 처리 중 오류가 발생했습니다: {str(e)}")

        except Exception as e:
            QMessageBox.warning(self, "오류", f"이미지 로드 중 오류가 발생했습니다: {str(e)}")

    def update_image_label(self, image):
        try:
            if image is None:
                return

            # RGB 형식 확인 및 변환
            if len(image.shape) == 3 and image.shape[2] == 3:
                if not isinstance(image, np.ndarray):
                    image = np.array(image)

                # BGR을 RGB로 변환
                if image.dtype != np.uint8:
                    image = image.astype(np.uint8)

                height, width, channel = image.shape
                bytes_per_line = 3 * width

                # QImage 생성 시 올바른 형식 지정
                q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)

                # 이미지 크기에 맞게 Label 조정
                scaled_pixmap = pixmap.scaled(
                    self.image_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )

                self.image_label.setPixmap(scaled_pixmap)
                self.image_label.setAlignment(Qt.AlignCenter)

        except Exception as e:
            print(f"Error updating image label: {e}")

    def display_objects_on_background(self):
        try:
            if self.background is None:
                print("No background image loaded.")
                return

            background = self.background.copy()

            for i, (img, pos) in enumerate(self.all_objects):
                try:
                    # 이미지 크기 조정
                    resized_image = cv2.resize(img, (100, 100), interpolation=cv2.INTER_LANCZOS4)

                    # 알파 채널 처리
                    if len(resized_image.shape) == 4:
                        alpha = resized_image[:, :, 3]
                        resized_image = resized_image[:, :, :3]
                        alpha = alpha / 255.0
                    else:
                        alpha = np.ones((resized_image.shape[0], resized_image.shape[1]))

                    x, y = pos

                    # 이미지가 배경 범위 내에 있는지 확인
                    if (y + resized_image.shape[0] <= background.shape[0] and
                            x + resized_image.shape[1] <= background.shape[1]):

                        # 알파 블렌딩
                        for c in range(3):
                            background[y:y + resized_image.shape[0],
                            x:x + resized_image.shape[1], c] = (
                                    alpha * resized_image[:, :, c] +
                                    (1 - alpha) * background[y:y + resized_image.shape[0],
                                                  x:x + resized_image.shape[1], c]
                            )

                except Exception as e:
                    print(f"Error processing object {i}: {e}")
                    continue

            self.update_image_label(background)

        except Exception as e:
            print(f"Error in display_objects_on_background: {e}")
    def mouse_event(self, event, x, y, flags, param):
        self.down = False
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_position = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            window_width, window_height = 640, 480
            image_height, image_width = self.background.shape[:2]
            self.down = True
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging_object_index != -1:
            dx, dy = self.drag_offset
            new_x = x - dx
            new_y = y - dy
            self.all_objects[self.dragging_object_index] = (
            self.all_objects[self.dragging_object_index][0], (new_x, new_y))
            self.display_objects_on_background()
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_object_index = -1

    def open_new_window3(self):
        if self.background is None or not self.all_objects:
            print("배경 이미지나 물체 이미지가 로드되지 않았습니다.")
            return

        frame_width, frame_height = 640, 480
        background = cv2.resize(self.background, (frame_width, frame_height))
        self.flip_states = [False] * len(self.all_objects)

        self.x_list = [pos[0] for _, pos in self.all_objects]
        self.y_list = [pos[1] for _, pos in self.all_objects]

        self.dx_list = [np.random.randint(1, 10) for _ in range(len(self.all_objects))]
        self.dy_list = [np.random.randint(1, 10) for _ in range(len(self.all_objects))]

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.out = cv2.VideoWriter('output.avi', fourcc, 30.0, (frame_width, frame_height))
        self.is_running = True

        frame_count = 0

        # 창을 전체 화면 모드로 설정하지 않고 크기 조정 가능하게 설정
        cv2.namedWindow("Floating Objects", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Floating Objects", frame_width, frame_height)
        cv2.setMouseCallback("Floating Objects", self.mouse_event)

        while self.is_running:
            self.frame = background.copy()

            for i, (img, pos) in enumerate(self.all_objects):
                try:
                    resized_image = cv2.resize(img, (100, 100))

                    if self.rotations[i] != 0:
                        M = cv2.getRotationMatrix2D((50, 50), self.rotations[i], 1)
                        resized_image = cv2.warpAffine(resized_image, M, (100, 100))

                    if resized_image.dtype != np.uint8:
                        resized_image = cv2.convertScaleAbs(resized_image)

                    if len(resized_image.shape) == 2:
                        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)

                    if len(resized_image.shape) == 3 and resized_image.shape[2] == 4:
                        mask = resized_image[:, :, 3]
                    else:
                        lower_white = np.array([200, 200, 200])
                        upper_white = np.array([255, 255, 255])
                        mask = cv2.inRange(resized_image, lower_white, upper_white)
                        mask = cv2.bitwise_not(mask)

                        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2BGRA)

                    resized_image[:, :, 3] = mask

                    self.x_list[i] += self.dx_list[i]
                    self.y_list[i] += self.dy_list[i]

                    if self.x_list[i] < 0:
                        self.x_list[i] = 0
                        self.dx_list[i] *= -1
                    elif self.x_list[i] + resized_image.shape[1] > self.frame.shape[1]:
                        self.x_list[i] = self.frame.shape[1] - resized_image.shape[1]
                        self.dx_list[i] *= -1
                    if self.y_list[i] < 0:
                        self.y_list[i] = 0
                        self.dy_list[i] *= -1
                    elif self.y_list[i] + resized_image.shape[0] > self.frame.shape[0]:
                        self.y_list[i] = self.frame.shape[0] - resized_image.shape[0]
                        self.dy_list[i] *= -1

                    if self.mouse_position:
                        mx, my = self.mouse_position
                        if self.x_list[i] <= mx <= self.x_list[i] + 100 and self.y_list[i] <= my <= self.y_list[
                            i] + 100:
                            self.dx_list[i] = np.random.randint(-10, 10)
                            self.dy_list[i] = np.random.randint(-10, 10)

                        if self.down == True:
                            if self.x_list[i] <= mx <= self.x_list[i] + 100 and self.y_list[i] <= my <= self.y_list[
                                i] + 100:
                                self.rotations[i] = (self.rotations[i] + 90) % 360
                                self.display_objects_on_background()
                                break

                    alpha_s = resized_image[:, :, 3] / 255.0
                    alpha_l = 1.0 - alpha_s

                    for c in range(0, 3):
                        self.frame[self.y_list[i]:self.y_list[i] + resized_image.shape[0],
                        self.x_list[i]:self.x_list[i] + resized_image.shape[1], c] = (
                                alpha_s * resized_image[:, :, c] + alpha_l * self.frame[self.y_list[i]:self.y_list[i] +
                                                                                                       resized_image.shape[
                                                                                                           0],
                                                                             self.x_list[i]:self.x_list[i] +
                                                                                            resized_image.shape[1], c])

                except cv2.error as e:
                    print(f"Error processing image at position {pos}: {e}")

            if self.text_info:
                self.frame = self.add_text_to_frame(self.frame, self.text_info)

            cv2.imshow('Floating Objects', self.frame)

            self.out.write(self.frame)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                self.is_running = False

            frame_count += 1

        self.out.release()
        cv2.destroyAllWindows()

    import shutil

    def open_new_window6(self):
        try:
            self.final_running = True
            if hasattr(sys, '_MEIPASS'):
                base_path = sys._MEIPASS
            else:
                base_path = os.path.dirname(os.path.abspath(__file__))

            fname = QFileDialog.getOpenFileName(self, "Open File", "./")
            if fname[0]:
                print(fname[0])
                file_path = urllib.parse.quote(fname[0])
                file_path = urllib.parse.unquote(file_path)
                video1 = cv2.VideoCapture(file_path)
                video2 = cv2.VideoCapture(file_path)
                video3 = cv2.VideoCapture(file_path)
                video4 = cv2.VideoCapture(file_path)

                if not video1.isOpened() or not video2.isOpened() or not video3.isOpened() or not video4.isOpened():
                    print("Error: One or more video files could not be opened.")
                    return

                self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                self.out = cv2.VideoWriter('result_final.avi', self.fourcc, 20.0, (1600, 1600))

                if not self.out.isOpened():
                    print("Error: Could not open output file.")
                    return

                while self.final_running:
                    ret1, frame1 = video1.read()
                    ret2, frame2 = video2.read()
                    ret3, frame3 = video3.read()
                    ret4, frame4 = video4.read()

                    if not ret1 or not ret2 or not ret3 or not ret4:
                        break

                    frame1 = cv2.resize(frame1, (800, 800))
                    frame1 = frame1[500:, 400:600, :]
                    frame1 = cv2.resize(frame1, (800, 800))

                    frame4 = cv2.resize(frame4, (800, 800))
                    middle = frame4[120:500, 450:600, :]
                    frame4 = cv2.resize(middle, (800, 800))

                    frame2 = cv2.resize(frame2, (800, 800))
                    frame2 = frame2[0:500, :450, :]
                    frame2 = cv2.resize(frame2, (800, 800))

                    frame3 = cv2.resize(frame3, (800, 800))
                    right = frame3[10:510, 620:, :]
                    frame3 = cv2.resize(right, (800, 800))

                    rows, cols, _ = frame1.shape

                    pts1 = np.float32([[0, 0], [0, rows], [cols, 0], [cols, rows]])
                    pts2 = np.float32([[300, 250], [10, rows - 50], [cols - 300, 250], [cols - 10, rows - 50]])
                    pts3 = np.float32([[0, 0], [0, rows], [cols, 0], [cols, rows]])
                    pts4 = np.float32([[546, 9], [500, 250], [780, 11], [788, 750]])

                    pts5 = np.float32([[0, 0], [0, rows], [cols, 0], [cols, rows]])
                    pts6 = np.float32([[6, 12], [14, 742], [256, 4], [302, 248]])

                    pts7 = np.float32([[0, 0], [0, rows], [cols, 0], [cols, rows]])
                    pts8 = np.float32([[251, 4], [306, 250], [550, 7], [505, 247]])

                    mtrx1 = cv2.getPerspectiveTransform(pts1, pts2)
                    dst1 = cv2.warpPerspective(frame1, mtrx1, (cols, rows))

                    mtrx2 = cv2.getPerspectiveTransform(pts3, pts4)
                    dst2 = cv2.warpPerspective(frame3, mtrx2, (cols, rows))

                    dst2 = cv2.resize(dst2, (cols, rows))

                    combined_frame = cv2.add(dst1, dst2)

                    mtrx3 = cv2.getPerspectiveTransform(pts5, pts6)
                    dst3 = cv2.warpPerspective(frame2, mtrx3, (cols, rows))

                    combined_frame = cv2.add(combined_frame, dst3)

                    mtrx4 = cv2.getPerspectiveTransform(pts7, pts8)
                    dst4 = cv2.warpPerspective(frame4, mtrx4, (cols, rows))

                    combined_frame = cv2.add(combined_frame, dst4)

                    combined_frame = cv2.resize(combined_frame, (1600, 1600))

                    cv2.imshow("합성된 영상", combined_frame)

                    self.out.write(combined_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                video1.release()
                video2.release()
                video3.release()
                video4.release()
                self.out.release()
                cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error: {e}")

    def open_new_window4(self):
        try:
            if hasattr(self, 'is_running'):
                if self.is_running == True:
                    self.is_running = False
                    try:
                        # XVID 코덱으로 변경하여 새로운 파일로 저장
                        save_path, _ = QFileDialog.getSaveFileName(
                            self,
                            "저장할 경로 선택",
                            "",
                            "MP4 Files (*.mp4);;AVI Files (*.avi);;All Files (*)"
                        )

                        if save_path:
                            # 현재 녹화 중인 파일을 닫습니다
                            if hasattr(self, 'out'):
                                self.out.release()

                            # 파일 확장자 확인
                            _, ext = os.path.splitext(save_path)
                            if ext.lower() == '.mp4':
                                # MP4 포맷으로 저장
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            else:
                                # AVI 포맷으로 저장 (XVID 코덱)
                                fourcc = cv2.VideoWriter_fourcc(*'XVID')

                            # 임시 파일을 읽어서 새로운 코덱으로 저장
                            temp_video = cv2.VideoCapture('output.avi')
                            fps = temp_video.get(cv2.CAP_PROP_FPS)
                            width = int(temp_video.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(temp_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

                            out_final = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

                            while True:
                                ret, frame = temp_video.read()
                                if not ret:
                                    break
                                out_final.write(frame)

                            temp_video.release()
                            out_final.release()

                            # 임시 파일 삭제
                            if os.path.exists('output.avi'):
                                os.remove('output.avi')

                            QMessageBox.information(self, "성공", "영상이 성공적으로 저장되었습니다!")
                        else:
                            if hasattr(self, 'out'):
                                self.out.release()
                            if os.path.exists('output.avi'):
                                os.remove('output.avi')

                    except Exception as e:
                        QMessageBox.warning(self, "경고", f"영상 저장 중 오류가 발생했습니다: {str(e)}")
                    finally:
                        if hasattr(self, 'out'):
                            self.out.release()
                        cv2.destroyAllWindows()

            if hasattr(self, 'final_running'):
                if self.final_running == True:
                    self.final_running = False
                    try:
                        save_path, _ = QFileDialog.getSaveFileName(
                            self,
                            "저장할 경로 선택",
                            "",
                            "MP4 Files (*.mp4);;AVI Files (*.avi);;All Files (*)"
                        )

                        if save_path:
                            if hasattr(self, 'out'):
                                self.out.release()

                            # 파일 확장자 확인
                            _, ext = os.path.splitext(save_path)
                            if ext.lower() == '.mp4':
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            else:
                                fourcc = cv2.VideoWriter_fourcc(*'XVID')

                            # 임시 파일을 읽어서 새로운 코덱으로 저장
                            temp_video = cv2.VideoCapture('result_final.avi')
                            fps = temp_video.get(cv2.CAP_PROP_FPS)
                            width = int(temp_video.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(temp_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

                            out_final = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

                            while True:
                                ret, frame = temp_video.read()
                                if not ret:
                                    break
                                out_final.write(frame)

                            temp_video.release()
                            out_final.release()

                            if os.path.exists('result_final.avi'):
                                os.remove('result_final.avi')

                            QMessageBox.information(self, "성공", "영상이 성공적으로 저장되었습니다!")
                        else:
                            if hasattr(self, 'out'):
                                self.out.release()
                            if os.path.exists('result_final.avi'):
                                os.remove('result_final.avi')

                    except Exception as e:
                        QMessageBox.warning(self, "경고", f"영상 저장 중 오류가 발생했습니다: {str(e)}")
                    finally:
                        if hasattr(self, 'out'):
                            self.out.release()
                        cv2.destroyAllWindows()

        except Exception as e:
            QMessageBox.warning(self, "오류", f"예상치 못한 오류가 발생했습니다: {str(e)}")
            if hasattr(self, 'out'):
                self.out.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    startWin = StartWindow()
    sys.exit(app.exec_())
