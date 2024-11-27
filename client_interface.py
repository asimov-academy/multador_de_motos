# client_interface.py

import os
import cv2
import numpy as np

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage

class InfractionsList(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.list_widget = QListWidget()
        self.layout.addWidget(self.list_widget)
        self.setLayout(self.layout)

    def add_infraction(self, image, placa, data_hora):
        item = QListWidgetItem()
        item_widget = QWidget()
        item_layout = QHBoxLayout()

        # Image label
        image_label = QLabel()
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        image_label.setPixmap(pixmap.scaled(200, 150, Qt.KeepAspectRatio))

        # Text label
        text_label = QLabel(f"Placa: {placa}\nData/Hora: {data_hora}")

        item_layout.addWidget(image_label)
        item_layout.addWidget(text_label)
        item_widget.setLayout(item_layout)
        item.setSizeHint(item_widget.sizeHint())

        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, item_widget)

class MainWindow(QMainWindow):
    # Define signals
    update_frame_signal = pyqtSignal(np.ndarray)
    update_infractions_signal = pyqtSignal(object, object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detecção de Infrações de Trânsito")

        video_width = 1280
        sidebar_width = 420
        width = video_width + sidebar_width
        self.setGeometry(100, 100, int(width), 720)  # Set window size to 720p

        # Widgets
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(video_width, 720)

        self.infractions_list = InfractionsList()
        self.infractions_list.setFixedWidth(sidebar_width)

        # Layouts
        main_layout = QHBoxLayout()
        video_layout = QVBoxLayout()
        video_layout.addWidget(self.video_label)
        main_layout.addLayout(video_layout)
        main_layout.addWidget(self.infractions_list)

        # Adjust proportions
        main_layout.setStretch(0, 3)  # 75% for video
        main_layout.setStretch(1, 1)  # 25% for sidebar

        # Central Widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Connect signals to slots
        self.update_frame_signal.connect(self.update_frame)
        self.update_infractions_signal.connect(self.update_infractions)
        self.processed_infractions = set()

    def update_frame(self, frame):
        if frame is not None and frame.size > 0:
            # Resize the frame to 1280x720
            frame = cv2.resize(frame, (1280, 720))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.video_label.setPixmap(pixmap)
        else:
            # Clear the video if the frame is invalid
            self.video_label.clear()

    def update_infractions(self, frame_moto_list, infracoes_detectadas):
        for infracao, frame_moto in zip(infracoes_detectadas, frame_moto_list):
            infracao_id = infracao['id_moto']
            if infracao_id not in self.processed_infractions:
                if frame_moto is None:
                    continue  # Skip if the image is not found
                imagem = cv2.cvtColor(frame_moto, cv2.COLOR_BGR2RGB)
                placa = infracao.get('placa', 'N/A')
                data_hora = infracao.get('tempo_infracao', 'N/A')
                self.infractions_list.add_infraction(imagem, placa, data_hora)
                self.processed_infractions.add(infracao_id)

    def closeEvent(self, event):
        event.accept()
