import sys
import cv2
import time
import threading
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize
from PyQt5.QtGui import QPixmap, QImage
from ultralytics import YOLO
from datetime import datetime
from config import (INPUT_VIDEO_PATH, YOLO_MODEL_PATH)
from video_processing import processar_frame
from object_tracking import inicializar_rastreadores
from utils import obter_propriedades_video
import numpy as np

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

        # Label da imagem
        image_label = QLabel()
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        image_label.setPixmap(pixmap.scaled(200, 150, Qt.KeepAspectRatio))

        # Label do texto
        text_label = QLabel(f"Placa: {placa}\nData/Hora: {data_hora}")

        item_layout.addWidget(image_label)
        item_layout.addWidget(text_label)
        item_widget.setLayout(item_layout)
        item.setSizeHint(item_widget.sizeHint())

        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, item_widget)

class VideoThread(QThread):
    frame_processed = pyqtSignal(np.ndarray)

    def __init__(self, arquivo_entrada):
        super().__init__()
        self.cap = cv2.VideoCapture(arquivo_entrada)
        self.model = YOLO(YOLO_MODEL_PATH)
        self.running = True
        self.trackers = inicializar_rastreadores()
        self.estado_semaforo = ['NOT RED']
        self.posicao_linha_parada = [None]
        self.infracoes_detectadas = []
        self.frame_count = 0
        self.semaforo_mais_alto = {'dados': None}
        self.unrecognized_plate_counter = [1]
        _, _, self.fps = obter_propriedades_video(self.cap)
        self.fps = max(1, self.fps)
        self.frame_duration = 1 / self.fps
        self.infraction_frame = None
        self.infraction_frame_time = 0
        self.showing_infraction = False
        self.overlay_duration = 15  # Duração em segundos

    def run(self):
        while self.running:
            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_original = frame.copy()

            frame_anotado, detected_moto, detected_semaforo = processar_frame(
                frame, self.model, self.trackers, self.estado_semaforo, self.posicao_linha_parada,
                self.infracoes_detectadas, self.frame_count, self.fps, self.semaforo_mais_alto,
                frame_original, INPUT_VIDEO_PATH,
                self.unrecognized_plate_counter
            )

            # Verificar se há uma nova infração detectada
            if self.infracoes_detectadas:
                ultima_infracao = self.infracoes_detectadas[-1]
                infracao_tempo = ultima_infracao.get('tempo_infracao_frame', None)
                if infracao_tempo == self.frame_count:
                    # Salvar o frame da infração para exibir por 15 segundos
                    self.infraction_frame = frame_anotado.copy()
                    self.infraction_frame_time = time.time()
                    self.showing_infraction = True

            # Se houver um frame de infração recente e dentro do período de overlay
            if self.showing_infraction and (time.time() - self.infraction_frame_time < self.overlay_duration):
                frame_to_emit = self.infraction_frame
            else:
                frame_to_emit = frame_anotado
                self.showing_infraction = False

            self.frame_processed.emit(frame_to_emit)
            self.frame_count += 1

            # Controlar o tempo para corresponder ao FPS do vídeo
            elapsed_time = time.time() - start_time
            sleep_time = max(0, self.frame_duration - elapsed_time)
            time.sleep(sleep_time)

    def stop(self):
        self.running = False
        self.cap.release()
        self.quit()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detecção de Infrações de Trânsito")
        self.setGeometry(100, 100, 1280, 720)  # Definir tamanho da janela para 720p

        # Widgets
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(960, 720)  # 75% de 1280 é 960, então vídeo é 960x720

        self.infractions_list = InfractionsList()
        self.infractions_list.setFixedWidth(320)  # 25% de 1280 é 320

        # Layouts
        main_layout = QHBoxLayout()
        video_layout = QVBoxLayout()
        video_layout.addWidget(self.video_label)
        main_layout.addLayout(video_layout)
        main_layout.addWidget(self.infractions_list)

        # Ajustar proporções
        main_layout.setStretch(0, 3)  # Proporção de 75% para o vídeo
        main_layout.setStretch(1, 1)  # Proporção de 25% para a sidebar

        # Central Widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Video Thread
        self.video_thread = VideoThread(INPUT_VIDEO_PATH)
        self.video_thread.frame_processed.connect(self.update_frame)
        self.video_thread.start()

        # Atualizar lista de infrações periodicamente
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_infractions)
        self.update_timer.start(1000)  # A cada 1 segundo

        self.processed_infractions = set()

    def update_frame(self, frame):
        if frame is not None and frame.size > 0:
            # Redimensionar o frame para 960x720 (75% de 1280x720)
            frame = cv2.resize(frame, (960, 720))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.video_label.setPixmap(pixmap)
        else:
            # Caso o frame seja inválido, limpar o vídeo
            self.video_label.clear()

    def update_infractions(self):
        infracoes = self.video_thread.infracoes_detectadas
        for infracao in infracoes:
            infracao_id = infracao['id_moto']
            if infracao_id not in self.processed_infractions:
                # Carregar a imagem da moto
                caminho_imagem = os.path.join(infracao['pasta'], f'motocicleta_{infracao_id}.jpg')
                imagem = cv2.imread(caminho_imagem)
                if imagem is None:
                    continue  # Se a imagem não for encontrada, pular

                imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

                placa = infracao['placa']
                data_hora = infracao['tempo_infracao']

                self.infractions_list.add_infraction(imagem, placa, data_hora)
                self.processed_infractions.add(infracao_id)

    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
