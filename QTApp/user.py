import math
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
        self.btnE_styleSheet = "QPushButton { border-radius: 10px; background-color: #4A5D75; color: white; text-align: center; height: 60px; width: 150px; }; QPushButton:hover:!pressed {background-color: ##155a73}"
        "QPushButton:active { border-radius: 30px; background-color: #4A5D75; color: white; text-align: center; height: 60px; width: 150px;}"
        self.btnD_styleSheet = "QPushButton:disabled { border-radius: 10px; background-color: #B5B5B5; color: #2F3E4F; text-align: center; height: 60px; width: 150px; }"

        label = QLabel()
        pixmap = QPixmap('bear.png')
        pixmap = pixmap .scaledToHeight(256)
        label.setPixmap(pixmap)
        label.resize(80, 80)
        self.dirlist = ""

        self.btn_DownloadImages = QPushButton('Загрузить фото', self)
        self.btn_DownloadImages.setStyleSheet(self.btnE_styleSheet)
        self.btn_DownloadImages.resize(100, 40)
        self.btn_DownloadImages.clicked.connect(self.load_images)

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

        self.pbar = QProgressBar(self)

        # setting its geometry
        self.pbar.setGeometry(30, 40, 200, 25)
        self.pbar.setFixedWidth(256)
        self.pbar.setVisible(False)

        self.gridbox = QGridLayout()
        self.gridbox.setAlignment(Qt.AlignCenter)
        self.gridbox.addItem(QSpacerItem(10, 10), 0, 0)

        self.gridbox.addWidget(label, 1, 1)
        self.gridbox.addItem(QSpacerItem(10, 10), 2, 1)
        self.gridbox.addWidget(self.btn_DownloadImages, 3, 1)
        self.gridbox.addItem(QSpacerItem(10, 10), 4, 1)
        self.gridbox.addWidget(self.btn_FindBears, 5, 1)

        self.gridbox.addItem(QSpacerItem(10, 10), 6, 2)
        self.gridbox.spacing()
        
        self.setLayout(self.gridbox)
        self.setStyleSheet(layoutStyleSheet)

    def load_images(self):
        # fileName = QFileDialog.getOpenFileName(self, "Выберите папку с изображениями", "/home/jana", "Image Files (*.png *.jpg *.bmp)")
        dirlist = QFileDialog.getExistingDirectory(None,"Выбрать папку",".")
        if dirlist == "": return

        self.btn_FindBears.setEnabled(True)
        self.btn_FindBears.setStyleSheet(self.btnE_styleSheet)
        self.dirlist = dirlist

    def test_ended(self):
        self.gridbox.addWidget(self.btn_DownloadImages, 2, 1)
        self.gridbox.addWidget(self.btn_FindBears, 3, 1)

        self.btn_FindBears.setEnabled(True)
        self.btn_FindBears.setVisible(True)

        self.btn_DownloadImages.setEnabled(True)
        self.btn_DownloadImages.setVisible(True)

        self.pbar.setVisible(False)
        self.ll.setVisible(False)

        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText("Полученные данные сохранены в файл \"bears.csv\"")
        msgBox.setWindowTitle("Обработка окончена!")
        msgBox.exec()

    def find_bears(self):
        if self.dirlist == "": return
        self.btn_FindBears.setEnabled(False)
        self.btn_FindBears.setVisible(False)

        self.btn_DownloadImages.setEnabled(False)
        self.btn_DownloadImages.setVisible(False)

        self.ll = QLabel(self)
        self.ll.setText("Прогресс поиска медведей")
        self.gridbox.addWidget(self.ll, 2, 1, Qt.AlignCenter)
        self.gridbox.addWidget(self.pbar, 3, 1, Qt.AlignCenter)
        self.pbar.setVisible(True)
        qApp.processEvents()
        # self.btn_SaveResults.setEnabled(True)
        # self.btn_SaveResults.setVisible(True)
        # add progress bar here 
        test_model(Path(self.dirlist), self.upd_pb, self.test_ended)
        pass

    def upd_pb(self, i):
        self.pbar.setValue(math.trunc(i * 100))
        qApp.processEvents()

    def save_results(self):
        dirlist = QFileDialog.getExistingDirectory(None, "Выбрать папку", ".")
        print(dirlist)
        pass


def run_app():
    app = QApplication(sys.argv)
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("wb_0.0.1")
    app.setWindowIcon(QIcon('bear.png'))
    mw = QMainWindow()
    mw.setMinimumSize(QSize(1000, 700))
    mw.setWindowTitle("Программа поиска белых медведей")
    main_ui = MainUI()
    mw.setCentralWidget(main_ui)

    mw.show()
    sys.exit(app.exec_())