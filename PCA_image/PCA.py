import cv2
import numpy as np
from sklearn.decomposition import PCA
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QLineEdit, \
    QHBoxLayout, QMessageBox, QMainWindow, QTabWidget
from PyQt5.QtGui import QPixmap, QImage, QIcon, QIntValidator


class PCAGui(QWidget):
    def __init__(self):
        super().__init__()  # Phuong thuc khoi tao cua lop co so QWidget

        # title cho cua so
        self.setWindowTitle("NÉN ẢNH PCA")
        self.icon = QIcon(".\\icon.png")
        self.setWindowIcon(self.icon)
        self.setGeometry(100, 100, 300, 100)  # kich thuoc cua so

        # --------------------------------------------------GIAO DIEN-------------------------------------------------#

        # Giao dien tab Image
        self.image_layout = QHBoxLayout()  # layout de hien thi 2 anh theo chieu ngang
        self.main_layout = QVBoxLayout()  # layout hien thi cac button, entry theo chieu doc

        self.label = QLabel()
        self.label1 = QLabel()
        self.label2 = QLabel()
        self.label3 = QLabel()
        self.label2.setText("Nhập số lượng thành phần chính: ")
        self.image_layout.addWidget(self.label1)  # label chua anh goc
        self.image_layout.addWidget(self.label)  # label chua anh sau khi PCA
        self.main_layout.addWidget(self.label2)
        self.main_layout.addWidget(self.label3)

        self.btnselect = QPushButton("Chọn ảnh")  # Button chon anh
        self.btnpca = QPushButton("PCA")  # Button PCA anh
        self.btnsave = QPushButton("Lưu ảnh")  # Button lưu ảnh PCA
        self.line_edit = QLineEdit()  # O nhap so luong thanh phan chinh
        self.line_edit.setValidator(QIntValidator(0, 999999999))  # gioi han gia tri nhap vào
        self.main_layout.addWidget(self.line_edit)
        self.main_layout.addWidget(self.btnselect)
        self.main_layout.addWidget(self.btnpca)
        self.main_layout.addWidget(self.btnsave)

        self.btnselect.clicked.connect(self.load_image)
        self.btnpca.clicked.connect(self.PCA_Image)
        self.btnsave.clicked.connect(self.save_ImagePCA)
        self.main_layout.addLayout(self.image_layout)
        self.setLayout(self.main_layout)

        # Message thong bao
        self.error = QMessageBox()
        self.error.setWindowTitle("Lỗi")
        self.error.setIcon(QMessageBox.Warning)

        self.succ = QMessageBox()
        self.succ.setWindowTitle("Thông báo")
        self.succ.setIcon(QMessageBox.Information)

        # Bien luu tru anh
        self.images = None  # bien chua anh dau vao
        self.img_uint8 = None  # bien chua anh da pca va dua ve dang uint8

    def load_image(self):
        # Hien thi hop thoai chon tep
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")

        if file_dialog.exec_():  # Kiem tra nguoi dung da chon tep hay chua
            file_path = file_dialog.selectedFiles()[0]
            pixmap1 = QPixmap(file_path)  # Doc va hien thi anh
            self.label1.setPixmap(pixmap1.scaled(500, 400))

            # Đọc ảnh từ tệp
            self.images = cv2.imread(file_path)
            self.label3.setText("Số thành phần chính tối đa của ảnh là: " + str(min(self.images.shape[0], self.images.shape[1])))
            print(self.images.shape)


    def PCA_Image(self):
        if self.images is not None:
            # lay gia tri tu entry
            text = self.line_edit.text()
            n_samples, n_features, _ = self.images.shape
            min_components = min(n_samples, n_features)

            if text.strip() == '':
                self.error.setText("Bạn vui lòng nhập số lượng thành phần chính!")
                self.error.exec_()
                return

            if int(text) > min_components or int(text) <= 0:
                self.error.setText("Số thành phần chính không hợp lệ, giá trị phải nằm giữa 0 và " + str(min_components))
                self.error.exec_()

            else:
                text_int = int(text)

                # Tach anh ra 3 kenh mau rieng biet
                B, G, R = cv2.split(self.images)

                # Chuan hoa ve khoang gia tri 0 den 1
                B_flat = B / 255
                G_flat = G / 255
                R_flat = R / 255

                # Khởi tạo đối tượng PCA cho từng kênh màu
                pca_B = PCA(n_components=text_int)
                pca_G = PCA(n_components=text_int)
                pca_R = PCA(n_components=text_int)

                # Thực hiện PCA cho từng kênh màu
                B_pca = pca_B.fit_transform(B_flat)
                G_pca = pca_G.fit_transform(G_flat)
                R_pca = pca_R.fit_transform(R_flat)
                print(B_pca.shape)

                # Chiếu lại ảnh gốc lên không gian PCA cho từng kênh màu
                B_pca = pca_B.inverse_transform(B_pca)
                G_pca = pca_G.inverse_transform(G_pca)
                R_pca = pca_R.inverse_transform(R_pca)
                print(B_pca.shape)

                # Kết hợp các kênh màu đã được PCA thành ảnh màu PCA
                pca_images = cv2.merge((B_pca, G_pca, R_pca))
                print(pca_images.dtype)

                img_min = np.min(pca_images)
                img_max = np.max(pca_images)
                self.img_uint8 = (((pca_images - img_min) / (img_max - img_min)) * 255).astype(np.uint8)
                print(self.img_uint8.dtype)

                h, w, c = self.img_uint8.shape
                bytes_per_line = c * w
                q_image = QImage(self.img_uint8.data, w, h, bytes_per_line, QImage.Format_BGR888)
                pixmap = QPixmap.fromImage(q_image)

                # Hiển thị ảnh lên giao diện người dùng
                self.label.setPixmap(pixmap.scaled(500, 400))

    def save_ImagePCA(self):
        if self.img_uint8 is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, 'Save Image', '', 'Image files (*.jpg)')

            if file_path:
                cv2.imwrite(file_path, self.img_uint8, [cv2.IMWRITE_JPEG_QUALITY, 70])
                self.succ.setText("Ảnh đã được lưu thành công")
                self.succ.exec_()


if __name__ == '__main__':
    app = QApplication([])
    window = PCAGui()
    window.show()
    app.exec_()
