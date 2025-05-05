from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from PyQt5 import QtCore, QtGui, QtWidgets
from sklearn.model_selection import cross_val_score, KFold

class Ui_MainWindow2(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1131, 782)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.model_gbox = QtWidgets.QGroupBox(self.centralwidget)
        self.model_gbox.setGeometry(QtCore.QRect(40, 10, 301, 161))
        self.model_gbox.setObjectName("model_gbox")
        self.logistic_cbox = QtWidgets.QRadioButton(self.model_gbox)
        self.logistic_cbox.setGeometry(QtCore.QRect(30, 40, 171, 20))
        self.logistic_cbox.setObjectName("logistic_cbox")
        self.rf_cbox = QtWidgets.QRadioButton(self.model_gbox)
        self.rf_cbox.setGeometry(QtCore.QRect(30, 80, 151, 20))
        self.rf_cbox.setObjectName("rf_cbox")
        self.knn_cbox = QtWidgets.QRadioButton(self.model_gbox)
        self.knn_cbox.setGeometry(QtCore.QRect(30, 120, 121, 20))
        self.knn_cbox.setObjectName("knn_cbox")
        self.normlizasyon_gbox = QtWidgets.QGroupBox(self.centralwidget)
        self.normlizasyon_gbox.setGeometry(QtCore.QRect(380, 20, 301, 161))
        self.model_button_group = QtWidgets.QButtonGroup(self.centralwidget)
        self.model_button_group.addButton(self.logistic_cbox)
        self.model_button_group.addButton(self.rf_cbox)
        self.model_button_group.addButton(self.knn_cbox)

        # Set exclusive to True (sadece birini seçebilirsiniz)
        self.model_button_group.setExclusive(True)

        self.normlizasyon_gbox.setObjectName("normlizasyon_gbox")
        self.zscore_cbox = QtWidgets.QCheckBox(self.normlizasyon_gbox)
        self.zscore_cbox.setGeometry(QtCore.QRect(20, 60, 201, 51))
        self.zscore_cbox.setObjectName("zscore_cbox")
        self.minmax_cbox = QtWidgets.QCheckBox(self.normlizasyon_gbox)
        self.minmax_cbox.setGeometry(QtCore.QRect(20, 20, 201, 51))
        self.minmax_cbox.setObjectName("minmax_cbox")
        self.normalizasyon_cbox = QtWidgets.QCheckBox(self.normlizasyon_gbox)
        self.normalizasyon_cbox.setGeometry(QtCore.QRect(20, 100, 201, 51))
        self.normalizasyon_cbox.setObjectName("normalizasyon_cbox")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(800, 620, 71, 20))
        self.normalize_button_group = QtWidgets.QButtonGroup(self.centralwidget)
        self.normalize_button_group.addButton(self.zscore_cbox)
        self.normalize_button_group.addButton(self.minmax_cbox)
        self.normalize_button_group.addButton(self.normalizasyon_cbox)

        # Set exclusive to True (sadece birini seçebilirsiniz)
        self.normalize_button_group.setExclusive(True)

        self.label_5.setObjectName("label_5")
        self.predict_gbox = QtWidgets.QGroupBox(self.centralwidget)
        self.predict_gbox.setGeometry(QtCore.QRect(390, 200, 291, 121))
        self.predict_gbox.setObjectName("predict_gbox")
        self.kullanma_cbox = QtWidgets.QCheckBox(self.predict_gbox)
        self.kullanma_cbox.setGeometry(QtCore.QRect(0, 80, 201, 51))
        self.kullanma_cbox.setObjectName("kullanma_cbox")
        self.predictprobe_cbox = QtWidgets.QCheckBox(self.predict_gbox)
        self.predictprobe_cbox.setGeometry(QtCore.QRect(0, 50, 201, 51))
        self.predictprobe_cbox.setObjectName("predictprobe_cbox")
        self.predict_cbox = QtWidgets.QCheckBox(self.predict_gbox)
        self.predict_cbox.setGeometry(QtCore.QRect(0, 20, 201, 51))
        self.predict_cbox.setObjectName("predict_cbox")
        self.acc = QtWidgets.QLineEdit(self.centralwidget)
        self.acc.setGeometry(QtCore.QRect(880, 520, 113, 22))
        self.predict_button_group = QtWidgets.QButtonGroup(self.centralwidget)
        self.predict_button_group.addButton(self.kullanma_cbox)
        self.predict_button_group.addButton(self.predict_cbox)
        self.predict_button_group.addButton(self.predictprobe_cbox)

        # Set exclusive to True (sadece birini seçebilirsiniz)
        self.predict_button_group.setExclusive(True)



        self.acc.setObjectName("acc")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(800, 670, 71, 20))
        self.label_6.setObjectName("label_6")
        self.kfold_gbox = QtWidgets.QGroupBox(self.centralwidget)
        self.kfold_gbox.setGeometry(QtCore.QRect(50, 190, 291, 111))
        self.kfold_gbox.setObjectName("kfold_gbox")
        self.radioButton = QtWidgets.QRadioButton(self.kfold_gbox)
        self.radioButton.setGeometry(QtCore.QRect(20, 30, 241, 20))
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(self.kfold_gbox)
        self.radioButton_2.setGeometry(QtCore.QRect(20, 70, 251, 20))
        self.radioButton_2.setObjectName("radioButton_2")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(840, -20, 101, 16))


        self.label.setObjectName("label")
        self.egitim_btn = QtWidgets.QPushButton(self.centralwidget)
        self.egitim_btn.setGeometry(QtCore.QRect(200, 340, 301, 41))
        self.egitim_btn.setObjectName("egitim_btn")
        self.matrix = QtWidgets.QTableWidget(self.centralwidget)
        self.matrix.setGeometry(QtCore.QRect(750, 340, 281, 131))
        self.matrix.setRowCount(2)
        self.matrix.setColumnCount(2)
        self.matrix.setObjectName("matrix")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(790, 520, 71, 20))
        self.label_3.setObjectName("label_3")
        self.spe = QtWidgets.QLineEdit(self.centralwidget)
        self.spe.setGeometry(QtCore.QRect(880, 620, 113, 22))
        self.spe.setObjectName("spe")
        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setGeometry(QtCore.QRect(750, 10, 281, 271))
        self.tableWidget.setRowCount(6)
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setObjectName("tableWidget")
        self.f1 = QtWidgets.QLineEdit(self.centralwidget)
        self.f1.setGeometry(QtCore.QRect(880, 670, 113, 20))
        self.f1.setObjectName("f1")
        self.recall = QtWidgets.QLineEdit(self.centralwidget)
        self.recall.setGeometry(QtCore.QRect(880, 570, 113, 22))
        self.recall.setObjectName("recall")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(820, 310, 121, 16))
        self.label_2.setObjectName("label_2")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(790, 570, 71, 20))
        self.label_4.setObjectName("label_4")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1131, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.kfold_acc_label = QtWidgets.QLabel(self.centralwidget)
        self.kfold_acc_label.setGeometry(QtCore.QRect(800, 720, 100, 20))
        self.kfold_acc_label.setObjectName("kfold_acc_label")
        self.kfold_acc_label.setText("K-Fold Acc")

        # K-Fold Accuracy Line Edit
        self.kfold_acc_lineedit = QtWidgets.QLineEdit(self.centralwidget)
        self.kfold_acc_lineedit.setGeometry(QtCore.QRect(880, 720, 113, 22))
        self.kfold_acc_lineedit.setObjectName("kfold_acc_lineedit")
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.new_table = QtWidgets.QTableWidget(self.centralwidget)
        self.new_table.setGeometry(QtCore.QRect(40, 400, 701, 100))  # Konum ve boyut
        self.new_table.setRowCount(1)  # 1 Satır
        self.new_table.setColumnCount(5)  # 5 Sütun

        # Sütun başlıklarını ayarla
        self.new_table.setHorizontalHeaderLabels(
            ["1. Doğrulama", "2. Doğrulama", "3. Doğrulama", "4. Doğrulama", "5. Doğrulama"])
        self.egitim_btn.clicked.connect(self.train_model)
        self.egitim_btn.setStyleSheet("background-color: green; color: white;")

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.model_gbox.setTitle(_translate("MainWindow", "EĞİTİM MODELİ SEÇİNİZ"))
        self.logistic_cbox.setText(_translate("MainWindow", "LOGİSTİC REGRESİON"))
        self.rf_cbox.setText(_translate("MainWindow", "RANDOM FOREST"))
        self.knn_cbox.setText(_translate("MainWindow", "KNN"))
        self.normlizasyon_gbox.setTitle(_translate("MainWindow", "Normalizasyon İşlemi "))
        self.zscore_cbox.setText(_translate("MainWindow", "Z-Score Normalizasyon"))
        self.minmax_cbox.setText(_translate("MainWindow", "Min-Max Normalizasyon"))
        self.normalizasyon_cbox.setText(_translate("MainWindow", "Normalizasyon yapmak istemiyorum"))
        self.label_5.setText(_translate("MainWindow", "ÖZGÜLLÜK"))
        self.predict_gbox.setTitle(_translate("MainWindow", "Predict-Predict Proba Kullanımı"))
        self.kullanma_cbox.setText(_translate("MainWindow", "Hiç birini kullanmak istemiyorum"))
        self.predictprobe_cbox.setText(_translate("MainWindow", "Predict Proba"))
        self.predict_cbox.setText(_translate("MainWindow", "Predict"))
        self.label_6.setText(_translate("MainWindow", "F1 SKOR"))
        self.kfold_gbox.setTitle(_translate("MainWindow", "K-FOLD ÇAPRAZ DOĞRULAMA"))
        self.radioButton.setText(_translate("MainWindow", "K-FOLD ÇAPRAZ DOĞRULAMA OLSUN"))
        self.radioButton_2.setText(_translate("MainWindow", "K-FOLD ÇAPRAZ DOĞRULAMA OLMASIN"))
        self.label.setText(_translate("MainWindow", "PREDİCT PROBA"))
        self.egitim_btn.setText(_translate("MainWindow", "Model Eğitimini Gerçekleştir ve Sonuçları Göster"))
        self.label_3.setText(_translate("MainWindow", "DORĞULUK"))
        self.label_2.setText(_translate("MainWindow", "KARIŞIKLIK MATRİSİ"))
        self.label_4.setText(_translate("MainWindow", "DUYARLILIK"))

    def train_model(self):
        # Veriyi seçme
        if self.zscore_cbox.isChecked():
            data = pd.read_excel("zskor_normalizasyon.xlsx")
        elif self.minmax_cbox.isChecked():
            data= pd.read_excel("dataset_normalizason.xlsx")
        elif self.normalizasyon_cbox.isChecked():
            data=pd.read_excel("dataset.xlsx")
        else:
            data=pd.read_excel("dataset.xlsx")

        # Özellikler ve hedefi ayırma
        X = data.drop(columns=['Result'])
        y = data['Result']

        # Veriyi ölçekleme
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Eğitim ve test verisi bölme
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if self.logistic_cbox.isChecked():
            model = LogisticRegression(max_iter=1000)
        elif self.rf_cbox.isChecked():
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.knn_cbox.isChecked():
            model = KNeighborsClassifier(n_neighbors=5)
        else:
            self.show_message("Hata", "Lütfen bir model seçin.")
            return

        # Modeli eğitme ve tahmin yapma
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Değerlendirme metriklerini hesaplama
        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

        if self.predict_cbox.isChecked():
            predictions = model.predict(X_test)
            # Tabloyu güncelleme işlemi
            for i, value in enumerate(predictions):
                self.tableWidget.setItem(i, 0, QtWidgets.QTableWidgetItem(f"Tahmin {i + 1}"))
                self.tableWidget.setItem(i, 1, QtWidgets.QTableWidgetItem(str(value)))
            # predict_proba kullanarak tahmin yapıyoruz
        elif self.predictprobe_cbox.isChecked():
            # İkinci sınıfın olasılıkları (1. sınıf olasılıkları için [:, 0] kullanılır)
            predictions_proba = model.predict_proba(X_test)[:, 1]  # İkinci sınıfın olasılıkları

            # Tabloyu güncelleme işlemi
            for i, value in enumerate(predictions_proba):
                self.tableWidget.setItem(i, 0, QtWidgets.QTableWidgetItem(f"Tahmin {i + 1}"))
                self.tableWidget.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{value:.2f}"))
        elif self.kullanma_cbox.isChecked():
            # Tabloyu temizleyelim
            self.tableWidget.clearContents()

            # "Tahmin yapılmadı" mesajını ekleyelim
            for i in range(6):  # 6 satır olduğu varsayımıyla
                self.tableWidget.setItem(i, 0, QtWidgets.QTableWidgetItem(f"Tahmin {i + 1}"))
                self.tableWidget.setItem(i, 1, QtWidgets.QTableWidgetItem("Tahmin yapılmadı"))

        matrix = cm  # Karışıklık matrisi
        self.matrix.clearContents()  # Önce tabloyu temizleyelim

        # Karışıklık matrisi sonuçlarını tabloya yerleştirme
        for i in range(2):  # Karışıklık matrisi genellikle 2x2 boyutunda (ikili sınıflandırma)
            for j in range(2):
                self.matrix.setItem(i, j, QtWidgets.QTableWidgetItem(str(matrix[i, j])))

        from PyQt5.QtWidgets import QMessageBox
        from PyQt5.QtWidgets import QTableWidgetItem


        try:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5 katmanlı çapraz doğrulama
            cv_scores = cross_val_score(model, X, y, cv=kf)
            average_accuracy = np.mean(cv_scores)

            if self.radioButton.isChecked():
                self.kfold_acc_lineedit.setText(f"{average_accuracy:.2%}")
                for i in range(5):
                    item = QTableWidgetItem(f"{cv_scores[i]:.2%}")  # Yüzde formatında
                    self.new_table.setItem(0, i, item)  # 0. satır ve i. sütun
            elif self.radioButton_2.isChecked():
                self.kfold_acc_lineedit.setText("NULL")

        except Exception as e:
            # Hata mesajı göster
            error_message = QMessageBox()
            error_message.setIcon(QMessageBox.Critical)
            error_message.setWindowTitle("Hata")
            error_message.setText("Bir hata oluştu!")
            error_message.setInformativeText(f"Hata mesajı: {str(e)}")
            error_message.exec_()

        # Arayüzdeki alanları güncelleme
        self.acc.setText(f"{acc:.2%}")
        self.recall.setText(f"{recall:.2%}")
        self.spe.setText(f"{specificity:.2%}")
        self.f1.setText(f"{f1:.2%}")




    # Kullanıcıya mesaj göstermek için show_message fonksiyonu
    def show_message(self, title, message):
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setIcon(QMessageBox.Warning)
        msg.exec_()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow2()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
