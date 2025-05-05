
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class Ui_MainWindow3(object):
    def setupUi(self, MainWindow3):
        MainWindow3.setObjectName("MainWindow3")
        MainWindow3.resize(1268, 829)
        self.centralwidget = QtWidgets.QWidget(MainWindow3)

        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(400, 50, 55, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(400, 120, 71, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(400, 180, 121, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(400, 230, 221, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.input_bs = QtWidgets.QLabel(self.centralwidget)
        self.input_bs.setGeometry(QtCore.QRect(400, 350, 161, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.input_bs.setFont(font)
        self.input_bs.setObjectName("input_bs")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(400, 420, 71, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(400, 290, 221, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(400, 470, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.input_age = QtWidgets.QLineEdit(self.centralwidget)
        self.input_age.setGeometry(QtCore.QRect(460, 60, 113, 22))
        self.input_age.setObjectName("input_age")
        self.input_gender = QtWidgets.QLineEdit(self.centralwidget)
        self.input_gender.setGeometry(QtCore.QRect(490, 120, 113, 22))
        self.input_gender.setObjectName("input_gender")
        self.input_hr = QtWidgets.QLineEdit(self.centralwidget)
        self.input_hr.setGeometry(QtCore.QRect(520, 180, 113, 22))
        self.input_hr.setObjectName("input_hr")
        self.input_sbp = QtWidgets.QLineEdit(self.centralwidget)
        self.input_sbp.setGeometry(QtCore.QRect(630, 230, 113, 22))
        self.input_sbp.setObjectName("input_sbp")
        self.input_dbp = QtWidgets.QLineEdit(self.centralwidget)
        self.input_dbp.setGeometry(QtCore.QRect(650, 290, 113, 22))
        self.input_dbp.setObjectName("input_dbp")
        self.lineEdit_6 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_6.setGeometry(QtCore.QRect(530, 360, 113, 22))
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.input_ck = QtWidgets.QLineEdit(self.centralwidget)
        self.input_ck.setGeometry(QtCore.QRect(500, 420, 113, 22))
        self.input_ck.setObjectName("input_ck")
        self.input_tr = QtWidgets.QLineEdit(self.centralwidget)
        self.input_tr.setGeometry(QtCore.QRect(510, 480, 113, 22))
        self.input_tr.setObjectName("input_tr")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(610, 60, 151, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(630, 120, 251, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(660, 180, 151, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(770, 230, 151, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(790, 290, 151, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        self.label_14.setGeometry(QtCore.QRect(680, 360, 151, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        self.label_15.setGeometry(QtCore.QRect(640, 420, 151, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.label_16 = QtWidgets.QLabel(self.centralwidget)
        self.label_16.setGeometry(QtCore.QRect(650, 480, 151, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_16.setFont(font)
        self.label_16.setObjectName("label_16")
        self.result = QtWidgets.QLineEdit(self.centralwidget)
        self.result.setGeometry(QtCore.QRect(560, 560, 161, 41))
        self.result.setObjectName("result")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(460, 560, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        MainWindow3.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow3)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1268, 26))
        self.menubar.setObjectName("menubar")
        MainWindow3.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow3)
        self.statusbar.setObjectName("statusbar")
        MainWindow3.setStatusBar(self.statusbar)
        self.tahmin_buton = QtWidgets.QPushButton(self.centralwidget)
        self.tahmin_buton.setGeometry(QtCore.QRect(500, 650, 200, 100))
        self.tahmin_buton.setObjectName("tahmin_buton")
        self.retranslateUi(MainWindow3)
        QtCore.QMetaObject.connectSlotsByName(MainWindow3)
        self.tahmin_buton.clicked.connect(self.veri_al)
        self.result.setReadOnly(True)
        self.tahmin_buton.setStyleSheet("background-color: green; color: white;")

    def retranslateUi(self, MainWindow3):
        _translate = QtCore.QCoreApplication.translate
        MainWindow3.setWindowTitle(_translate("MainWindow3", "MainWindow"))
        self.label.setText(_translate("MainWindow3", "Age"))
        self.label_2.setText(_translate("MainWindow3", "Gender"))
        self.label_3.setText(_translate("MainWindow3", "Heart Rate"))
        self.label_4.setText(_translate("MainWindow3", "Systolic blood pressure"))
        self.input_bs.setText(_translate("MainWindow3", "Blood sugar"))
        self.label_6.setText(_translate("MainWindow3", "CK-MB"))
        self.label_7.setText(_translate("MainWindow3", "Diastolic blood pressure"))
        self.label_8.setText(_translate("MainWindow3", "Troponin"))
        #self.label_9.setText(_translate("MainWindow3", "0<VALUE<=1"))
        self.label_10.setText(_translate("MainWindow3", "ERKEK=1 ; KADIN=0"))
        self.label_11.setText(_translate("MainWindow3", "0<VALUE<=1"))
        self.label_12.setText(_translate("MainWindow3", "0<VALUE<=1"))
        self.label_13.setText(_translate("MainWindow3", "0<VALUE<=1"))
        self.label_14.setText(_translate("MainWindow3", "0<VALUE<=1"))
        self.label_15.setText(_translate("MainWindow3", "0<VALUE<=1"))
        self.label_16.setText(_translate("MainWindow3", "0<VALUE<=1"))
        self.label_5.setText(_translate("MainWindow3", "RESULT"))
        self.tahmin_buton.setText(_translate("MainWindow3", "SONUÇ İÇİN TIKLA"))

    def veri_al(self):
        try:
            # Kullanıcıdan alınan veriler
            age_data = float(self.input_age.text())
            gender_data = float(self.input_gender.text())
            hr_data = float(self.input_hr.text())
            sbp_data = float(self.input_sbp.text())
            dbp_data = float(self.input_dbp.text())
            bs_data = float(self.lineEdit_6.text())
            ck_data = float(self.input_ck.text())
            tr_data = float(self.input_tr.text())

            # Girilen değerlerin 0 < value <= 1 arasında olup olmadığını kontrol etme
            if not (0 < age_data <= 100):
                raise ValueError("Age değeri 0 ile 100 arasında olmalıdır.")
            if not (gender_data == 1 or gender_data==0):
                raise ValueError("Gender değeri 0 ya da 1  olmalıdır.")
            if not (0 < hr_data <= 1):
                raise ValueError("Heart Rate değeri 0 ile 1 arasında olmalıdır.")
            if not (0 < sbp_data <= 1):
                raise ValueError("Systolic Blood Pressure değeri 0 ile 1 arasında olmalıdır.")
            if not (0 < dbp_data <= 1):
                raise ValueError("Diastolic Blood Pressure değeri 0 ile 1 arasında olmalıdır.")
            if not (0 < bs_data <= 1):
                raise ValueError("Blood Sugar değeri 0 ile 1 arasında olmalıdır.")
            if not (0 < ck_data <= 1):
                raise ValueError("CK-MB değeri 0 ile 1 arasında olmalıdır.")
            if not (0 < tr_data <= 1):
                raise ValueError("Troponin değeri 0 ile 1 arasında olmalıdır.")

            # Veri setini yükleme ve temizleme
            file_path = "dataset_normalizasyon.xlsx"
            data = pd.read_excel(file_path)
            data_cleaned = data.fillna(0)

            # Model eğitimi
            X = data_cleaned.drop(columns=['Result'])
            y = data_cleaned['Result']
            X = pd.get_dummies(X, drop_first=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            rf_model = RandomForestClassifier(random_state=42)
            rf_model.fit(X_train, y_train)

            # Kullanıcıdan alınan veriyi uygun forma dönüştürme
            new_data = pd.DataFrame([{
                'Age': age_data,
                'Gender': gender_data,
                'Heart Rate': hr_data,
                'Systolic Blood Pressure': sbp_data,
                'Diastolic Blood Pressure': dbp_data,
                'Blood Sugar': bs_data,
                'CK-MB': ck_data,
                'Troponin': tr_data
            }])
            new_data = pd.get_dummies(new_data, drop_first=True).reindex(columns=X.columns, fill_value=0)

            # Tahmin işlemi
            prediction = rf_model.predict(new_data)[0]

            if str(prediction) == 1:
                self.result.setText("HASTA")
            else:
                self.result.setText("HASTA DEĞİL")

        except ValueError as e:
            QMessageBox.critical(None, "Hata", str(e))
        except Exception as e:
            QMessageBox.critical(None, "Hata", f"Bir hata oluştu: {e}")



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow3 = QtWidgets.QMainWindow()
    ui = Ui_MainWindow3()
    ui.setupUi(MainWindow3)
    MainWindow3.show()
    sys.exit(app.exec_())
