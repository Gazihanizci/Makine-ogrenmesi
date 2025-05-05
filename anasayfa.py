from PyQt5 import QtCore, QtGui, QtWidgets
from gurultu import Ui_MainWindow1
from orijinal import Ui_MainWindow2
from user import  Ui_MainWindow3
class Ui_MainWindow(object):
    def orgac(self):
        self.window=QtWidgets.QMainWindow()
        self.ui=Ui_MainWindow2()
        self.ui.setupUi(self.window)
        self.window.show()
        MainWindow.hide()
    def eksikac(self):
        self.window=QtWidgets.QMainWindow()
        self.ui=Ui_MainWindow1()
        self.ui.setupUi(self.window)
        self.window.show()
        MainWindow.hide()
    def uiac(self):
        self.window=QtWidgets.QMainWindow()
        self.ui=Ui_MainWindow3()
        self.ui.setupUi(self.window)
        self.window.show()
        MainWindow.hide()
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 800)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.orijinal = QtWidgets.QPushButton(self.centralwidget)
        self.orijinal.setGeometry(QtCore.QRect(210, 70, 421, 161))
        self.orijinal.setObjectName("orijinal")
        self.user = QtWidgets.QPushButton(self.centralwidget)
        self.user.setGeometry(QtCore.QRect(210, 450, 421, 161))
        self.user.setObjectName("user")
        self.eksik = QtWidgets.QPushButton(self.centralwidget)
        self.eksik.setGeometry(QtCore.QRect(210, 260, 421, 161))
        self.eksik.setObjectName("eksik")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.orijinal.clicked.connect(self.orgac)
        self.eksik.clicked.connect(self.eksikac)
        self.user.clicked.connect(self.uiac)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.label_tavsiye = QtWidgets.QLabel(self.centralwidget)
        self.label_tavsiye.setGeometry(QtCore.QRect(0, 0, 500, 100))  # Konum ve boyut
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_tavsiye.setFont(font)
        self.label_tavsiye.setWordWrap(True)  # Metnin satır atlamasını sağlamak için
        self.label_tavsiye.setObjectName("label_tavsiye")
        self.label_tavsiye.setText(
            "TAVSİYE EDİLEN: EKSİK VE GÜRÜLTÜLÜ VERİLERLE ÇALIŞMAK")
        self.eksik.setStyleSheet("background-color: green; color: white;")
        self.orijinal.setStyleSheet("background-color: green; color: white;")
        self.user.setStyleSheet("background-color: green; color: white;")

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.orijinal.setText(_translate("MainWindow", "ORİJİNAL VERİ SETİ İLE ÇALIŞMAK İÇİN TIKLAYINIZ."))
        self.eksik.setText(_translate("MainWindow", "EKSİK VE GÜRÜLTÜLÜ VERİ SETİ İLE ÇALIŞMAK İÇİN TIKLAYINIZ."))
        self.user.setText(_translate("MainWindow", "KULLANICI ARAYÜZÜ İÇİN TIKLAYINIZ."))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
