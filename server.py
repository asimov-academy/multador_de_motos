# server.py

import cv2
import numpy as np
import pytesseract
from utils import calcular_linha_parada, formatar_data_hora, adicionar_texto_utf8
from object_tracking import atualizar_rastreadores, inicializar_rastreadores
from infractions import salvar_infracao
from config import (
    CLASSES_OF_INTEREST, CONFIDENCE_THRESHOLD, SEMAFORO_ALTURA_REAL,
    DISTANCIA_SEMAFORO_LINHA_PARADA, TESSERACT_CMD, INFRACTIONS_FOLDER
)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from typing import Dict
from ultralytics import YOLO
import json
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
import os
import asyncio

app = FastAPI()
executor = None  # Global executor


global model
global trackers
model = YOLO('yolo/yolo11x.pt')
trackers = inicializar_rastreadores()

def init_process():
    global model
    global trackers
    model = YOLO('yolo/yolo11x.pt')
    trackers = inicializar_rastreadores()

# Class to maintain the state of each connection
class ConnectionState:
    def __init__(self):
        # Initialize per-connection state variables
        self.model = None
        self.trackers = (None, None)
        self.estado_semaforo = [None]
        self.posicao_linha_parada = [None]
        self.infracoes_detectadas = []
        self.frame_count = 0
        self.fps = 30  # Adjust as necessary
        self.semaforo_mais_alto = {'dados': None}
        self.unrecognized_plate_counter = [1]

    def initialize(self):
        # Initialize model and trackers
        self.trackers = inicializar_rastreadores()

executor = None  # Global executor

@asynccontextmanager
async def lifespan(app: FastAPI):
    global executor
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    # Create the ProcessPoolExecutor with initializer
    executor = ProcessPoolExecutor(max_workers=16, initializer=init_process)
    yield
    executor.shutdown()

# Connection manager to maintain the state of each client
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[WebSocket, ConnectionState] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        # Initialize per-connection state
        state = ConnectionState()
        state.initialize()
        self.active_connections[websocket] = state

    def disconnect(self, websocket: WebSocket):
        self.active_connections.pop(websocket, None)

    def get_state(self, websocket: WebSocket) -> ConnectionState:
        return self.active_connections.get(websocket)

manager = ConnectionManager()

# Event handler for application startup
@app.on_event("startup")
async def startup_event():
    # Configure Tesseract OCR if necessary
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    state = manager.get_state(websocket)
    try:
        loop = asyncio.get_running_loop()
        while True:
            # Receive data from client
            data = await websocket.receive_text()
            message = json.loads(data)

            # Extract necessary parameters
            frame_data = message.get('frame')
            frame_id = message.get('frame_id')
            frame_bytes = bytes.fromhex(frame_data)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            frame_original = frame.copy()

            # Process the frame
            frame_anotado, detected_moto, detected_semaforo, caminho_moto = await loop.run_in_executor(
                executor,
                processar_frame,
                frame,
                state.estado_semaforo,
                state.posicao_linha_parada,
                state.infracoes_detectadas,
                state.frame_count,
                state.fps,
                state.semaforo_mais_alto,
                'arquivo_entrada',  # Adjust as necessary
                state.unrecognized_plate_counter
            )
            state.frame_count += 1

            # Encode the annotated frame to send back
            _, buffer = cv2.imencode('.jpg', frame_anotado)
            frame_anotado_bytes = buffer.tobytes()
            frame_anotado_hex = frame_anotado_bytes.hex()

            # Prepare a list of frame_moto_hex corresponding to each infraction
            frame_moto_hex_list = []
            for infracao in state.infracoes_detectadas:
                caminho_moto = infracao.get('caminho_moto', '')
                if caminho_moto and os.path.exists(caminho_moto):
                    img_moto = cv2.imread(caminho_moto)
                    if img_moto is not None:
                        _, buffer = cv2.imencode('.jpg', img_moto)
                        frame_moto_bytes = buffer.tobytes()
                        frame_moto_hex = frame_moto_bytes.hex()
                        frame_moto_hex_list.append(frame_moto_hex)
                    else:
                        print(f"Failed to read motorcycle image from {caminho_moto}")
                        frame_moto_hex_list.append('')
                else:
                    frame_moto_hex_list.append('')

            response = {
                'frame_id': frame_id,
                'frame_anotado': frame_anotado_hex,
                'detected_moto': detected_moto,
                'detected_semaforo': detected_semaforo,
                'infracoes_detectadas': state.infracoes_detectadas,
                'frame_moto_hex_list': frame_moto_hex_list
            }

            # Send the response to the client
            await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Cliente desconectado")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
        await websocket.close()

# Function to process the frame
def processar_frame(
        frame, estado_semaforo, posicao_linha_parada,
        infracoes_detectadas, frame_count, fps, semaforo_mais_alto,
        arquivo_entrada, unrecognized_plate_counter
    ):
    global model
    global trackers
    frame_original = frame.copy()
    altura_frame, largura_frame, _ = frame.shape

    resultados = model(frame)

    # Map class names to numeric IDs
    class_name_to_id = {'motorcycle': 1, 'traffic light': 2, 'license plate': 3}

    detections = []

    detected_moto = False
    detected_semaforo = False

    for resultado in resultados:
        boxes = resultado.boxes
        for box in boxes:
            cls = int(box.cls[0])
            confianca = float(box.conf[0])
            rotulo = model.names[cls]

            if rotulo in CLASSES_OF_INTEREST and confianca > CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width = x2 - x1
                height = y2 - y1
                detection = [
                    [x1, y1, width, height],
                    confianca,
                    class_name_to_id[rotulo]
                ]
                detections.append(detection)

    # Separate detections
    det_motos = [det for det in detections if det[2] == class_name_to_id['motorcycle']]
    det_semaforos = [det for det in detections if det[2] == class_name_to_id['traffic light']]
    det_placas = [det for det in detections if det[2] == class_name_to_id['license plate']]

    if det_motos:
        detected_moto = True
    if det_semaforos:
        detected_semaforo = True

    tracker_motos, tracker_semaforos = trackers

    # Update trackers
    tracks_motos, tracks_semaforos = atualizar_rastreadores(
        tracker_motos, tracker_semaforos, det_motos, det_semaforos, frame
    )

    # Process traffic lights
    semaforos = []
    for track in tracks_semaforos:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        bbox = track.to_ltwh()
        x1, y1, w, h = map(int, bbox)
        x2, y2 = x1 + w, y1 + h
        posicao = ((x1 + x2) // 2, (y1 + y2) // 2)
        tamanho_pixels = h
        # Process the state of the traffic light
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Detect red color
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        red_pixels = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        if total_pixels == 0:
            continue
        red_percentage = red_pixels / total_pixels
        if red_percentage > 0.1:
            estado = 'RED'
            cor = (0, 0, 255)
            rotulo = f'Semáforo {track_id} (RED)'
        else:
            estado = 'NOT RED'
            cor = (0, 255, 255)
            rotulo = f'Semáforo {track_id} (NOT RED)'

        semaforo = {
            'id': track_id,
            'posicao': posicao,
            'tamanho_pixels': tamanho_pixels,
            'bbox': (x1, y1, x2, y2),
            'estado': estado
        }
        semaforos.append(semaforo)

        cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)

        adicionar_texto_utf8(
            frame,
            rotulo,
            (x1 + 20, y1 - 10),
            cor
        )

    # Select the highest traffic light in the frame
    if semaforos:
        semaforo_mais_alto_atual = min(semaforos, key=lambda s: s['posicao'][1])
        # Update the highest traffic light
        semaforo_mais_alto['dados'] = semaforo_mais_alto_atual
        estado_semaforo[0] = semaforo_mais_alto_atual['estado']
    else:
        semaforo_mais_alto['dados'] = None
        estado_semaforo[0] = 'NOT RED'

    # Calculate the position of the stop line
    if semaforo_mais_alto['dados'] is not None:
        posicao_semaforo = semaforo_mais_alto['dados']['posicao']
        tamanho_semaforo_pixels = semaforo_mais_alto['dados']['tamanho_pixels']
        posicao_linha_parada[0] = calcular_linha_parada(
            posicao_semaforo, tamanho_semaforo_pixels, altura_frame
        )
    else:
        if posicao_linha_parada[0] is None:
            posicao_linha_parada[0] = int(altura_frame * 0.75)  # Adjust as necessary

    # Draw the stop line
    cv2.line(frame, (0, posicao_linha_parada[0]),
             (largura_frame, posicao_linha_parada[0]), (255, 0, 0), 2)

    # Configure Tesseract OCR if necessary
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    # Process motorcycles
    caminho_moto = None  # Initialize caminho_moto
    for track in tracks_motos:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        bbox = track.to_ltwh()
        x1, y1, w, h = map(int, bbox)
        x2, y2 = x1 + w, y1 + h
        centroide = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Ensure posicao_linha_parada[0] is not None
        if posicao_linha_parada[0] is not None:
            passou_linha = centroide[1] < posicao_linha_parada[0]
        else:
            passou_linha = False

        # Check if the motorcycle already committed an infraction
        infracao_existente = next((inf for inf in infracoes_detectadas if inf['id_moto'] == track_id), None)

        if passou_linha and estado_semaforo[0] == 'RED':
            if infracao_existente is None:
                # Motorcycle crossed the stop line with red light
                # Extract plate (if available)
                placa_texto = 'placa_nao_identificada'
                # Search for plates within the motorcycle bounding box
                placa_encontrada = False
                for det in det_placas:
                    x1p, y1p, wp, hp = det[0]
                    x2p, y2p = x1p + wp, y1p + hp
                    # Check if the plate is within the motorcycle
                    if x1p >= x1 and y1p >= y1 and x2p <= x2 and y2p <= y2:
                        placa_roi = frame_original[y1p:y2p, x1p:x2p]
                        # Perform OCR on the plate
                        placa_texto_ocr = pytesseract.image_to_string(
                            placa_roi, config='--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                        )
                        placa_texto_ocr = placa_texto_ocr.strip().replace(' ', '')
                        if placa_texto_ocr:
                            placa_texto = placa_texto_ocr
                        else:
                            # Plate not recognized
                            placa_texto = f'placa_nao_reconhecida_{unrecognized_plate_counter[0]}'
                            unrecognized_plate_counter[0] += 1
                        placa_encontrada = True
                        break
                if not placa_encontrada:
                    # Plate not detected within the motorcycle
                    placa_texto = f'placa_nao_reconhecida_{unrecognized_plate_counter[0]}'
                    unrecognized_plate_counter[0] += 1

                # Save motorcycle image
                moto_roi = frame_original[y1:y2, x1:x2]
                # Save infraction in a separate folder
                caminho_moto = salvar_infracao(
                    moto_roi,
                    placa_texto,
                    frame_original,
                    (x1, y1, x2, y2),
                    infracoes_detectadas,
                    track_id,
                    semaforo_mais_alto,
                    posicao_linha_parada[0],
                    arquivo_entrada,
                    frame_count,  # Pass the current frame
                    fps
                )
                print(f"Infraction saved, caminho_moto: {caminho_moto}")
                # Add the caminho_moto to the last infraction detected
                infracoes_detectadas[-1]['caminho_moto'] = caminho_moto

                # Draw alert
                adicionar_texto_utf8(
                    frame,
                    f'INFRAÇÃO Moto {track_id}',
                    (x1 + 20, y1 - 20),
                    (0, 0, 255)
                )
            else:
                caminho_moto = None

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(frame, f'Moto {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Apply blur to detected plates
    aplicar_desfoque_placas(frame, det_placas)

    # Add date and time
    data_hora = formatar_data_hora()
    cv2.putText(frame, data_hora, (largura_frame - 300, altura_frame - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame, detected_moto, detected_semaforo, caminho_moto

def aplicar_desfoque_placas(frame, det_placas):
    """Applies blur to detected plates in the frame."""
    for det in det_placas:
        x1p, y1p, wp, hp = det[0]
        x2p, y2p = x1p + wp, y1p + hp

        # Ensure coordinates are within frame limits
        altura_frame, largura_frame, _ = frame.shape
        x1p = max(0, x1p)
        y1p = max(0, y1p)
        x2p = min(largura_frame, x2p)
        y2p = min(altura_frame, y2p)

        # Apply blur to the plate region
        roi = frame[y1p:y2p, x1p:x2p]
        if roi.size == 0:
            continue
        roi_blur = cv2.GaussianBlur(roi, (51, 51), 0)
        frame[y1p:y2p, x1p:x2p] = roi_blur

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
