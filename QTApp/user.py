import sys
import ctypes

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from pathlib import Path
from WBnn.test_model import test_model


class MainUI(QWidget):
    start_training = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.initUI()


    def initUI(self):

        layoutStyleSheet = "background-color: #white;"
        self.btnE_styleSheet = "QPushButton { border-radius: 10px; background-color: #4A5D75; color: white; text-align: center; height: 60px; width: 150px; };"
        "QPushButton:active { border-radius: 30px; background-color: #4A5D75; color: white; text-align: center; height: 60px; width: 150px;}"
        self.btnD_styleSheet = "QPushButton:disabled { border-radius: 10px; background-color: #B5B5B5; color: #2F3E4F; text-align: center; height: 60px; width: 150px; }"

        label = QLabel()
        pixmap = QPixmap('bear.png')
        label.setPixmap(pixmap)
        label.resize(80, 80)

        self.btn_DownloadImages = QPushButton('Загрузить фото', self)
        self.btn_DownloadImages.setStyleSheet(self.btnE_styleSheet)
        self.btn_DownloadImages.resize(100, 40)
        self.btn_DownloadImages.clicked.connect(self.load_images)
        self.start_training.connect(self.begin_training)

        self.btn_FindBears = QPushButton('Запустить поиск', self)
        self.btn_FindBears.setStyleSheet(self.btnD_styleSheet)
        self.btn_FindBears.setEnabled(False)
        self.btn_FindBears.resize(150, 50)
        self.btn_FindBears.clicked.connect(self.find_bears)

        self.btn_SaveResults = QPushButton('Сохранить результат в папку', self)
        self.btn_SaveResults.setEnabled(False)
        self.btn_SaveResults.setVisible(False)
        self.btn_SaveResults.setStyleSheet(self.btnE_styleSheet)
        self.btn_SaveResults.resize(150, 50)
        self.btn_SaveResults.clicked.connect(self.save_results)

        self.vbox = QVBoxLayout()
        self.vbox.setAlignment(Qt.AlignCenter)
        self.vbox.spacing()
        self.vbox.addWidget(label)
        self.vbox.addSpacing(20)
        self.vbox.addWidget(self.btn_DownloadImages)
        self.vbox.addSpacing(10)
        self.vbox.addWidget(self.btn_FindBears)
        self.vbox.spacing()
        
        self.setLayout(self.vbox)
        self.setStyleSheet(layoutStyleSheet)

    def load_images(self):
        # fileName = QFileDialog.getOpenFileName(self, "Выберите папку с изображениями", "/home/jana", "Image Files (*.png *.jpg *.bmp)")
        dirlist = QFileDialog.getExistingDirectory(None,"Выбрать папку",".")
        if dirlist == "": return

        self.btn_FindBears.setEnabled(True)
        self.btn_FindBears.setStyleSheet(self.btnE_styleSheet)
        self.start_training.emit(dirlist)

    @pyqtSlot(str)
    def begin_training(self, dirlist):
        print(111)
        test_model(Path(dirlist))

    def find_bears(self):
        self.btn_FindBears.setEnabled(False)
        self.btn_FindBears.setVisible(False)

        self.btn_DownloadImages.setEnabled(False)
        self.btn_DownloadImages.setVisible(False)

        self.vbox.addWidget(self.btn_SaveResults)
        self.btn_SaveResults.setEnabled(True)
        self.btn_SaveResults.setVisible(True)
        # add progress bar here 

        pass

    def save_results(self):
        dirlist = QFileDialog.getExistingDirectory(None, "Выбрать папку", ".")
        print(dirlist)
        pass


def run_app():
    app = QApplication(sys.argv)
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("wb_0.0.1")
    app.setWindowIcon(QIcon('bear.png'))
    mw = QMainWindow()
    mw.setMinimumSize(QSize(1500, 1000))
    mw.setWindowTitle("Программа поиска белых медведей")
    main_ui = MainUI()
    mw.setCentralWidget(main_ui)

    mw.show()
    sys.exit(app.exec_())