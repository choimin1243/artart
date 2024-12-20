import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QLineEdit, QFormLayout, QDialog
from PyQt5.QtGui import QFontDatabase, QFont


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

        layout.addWidget(GPT_API)
        layout.addWidget(Edit_button)
        layout.addWidget(start_button)
        layout.addWidget(AI_button)

        self.setLayout(layout)
        self.setWindowTitle('시작하기')
        self.resize(300, 200)
        self.apply_stylesheet()
        self.show()

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
            font-size: 20px;  # 폰트 크기를 너무 크게 설정하지 마세요
            border-radius: 10px;
        }}
        QPushButton:hover {{
            background-color: #0000CD;
        }}
        QPushButton:pressed {{
            background-color: #FF8C00;
        }}
        QLabel {{
            color: #333333;
            font-size: 16px;
        }}
        QLineEdit {{
            background-color: #FFFFFF;
            color: #333333;
            border: 2px solid #FFD700;
            padding: 12px;
            font-size: 18px;
            border-radius: 10px;
        }}
        """
        self.setStyleSheet(qss)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    startWin = StartWindow()
    sys.exit(app.exec_())
