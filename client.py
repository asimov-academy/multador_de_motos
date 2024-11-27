# client.py

import sys
import cv2
import asyncio
import websockets
import json
import numpy as np
import qasync

from PyQt5.QtWidgets import QApplication
from client_interface import MainWindow

async def send_frames(uri, video_path, window):
    async with websockets.connect(uri) as websocket:
        cap = cv2.VideoCapture(video_path)  # Use 0 for webcam or provide video path
        frame_id = 0
        running = True

        # Queues to manage sending and receiving
        send_queue = asyncio.Queue()
        receive_queue = asyncio.Queue()

        # Start send and receive tasks
        send_task = asyncio.create_task(send_frames_task(cap, websocket, send_queue, receive_queue, frame_id, running))
        receive_task = asyncio.create_task(receive_frames_task(websocket, receive_queue, running))
        process_task = asyncio.create_task(process_frames_task(receive_queue, window))

        await asyncio.gather(send_task, receive_task, process_task)

        cap.release()
        cv2.destroyAllWindows()

async def send_frames_task(cap, websocket, send_queue, receive_queue, frame_id, running):
    while cap.isOpened() and running:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % 3 == 0:
            continue

        # Encode frame to bytes and convert to hexadecimal
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        frame_hex = frame_bytes.hex()

        # Prepare the message to be sent
        message = {
            'frame_id': frame_id,
            'frame': frame_hex,
            # Other parameters if necessary
        }

        # Send message to the server
        await websocket.send(json.dumps(message))

        # Control the sending rate if necessary
        await asyncio.sleep(0)  # Adjust the value as needed

async def receive_frames_task(websocket, receive_queue, running):
    while running:
        try:
            response_data = await websocket.recv()
            response = json.loads(response_data)

            frame_id = response['frame_id']
            frame_anotado_hex = response['frame_anotado']
            detected_moto = response['detected_moto']
            detected_semaforo = response['detected_semaforo']
            infracoes_detectadas = response['infracoes_detectadas']
            frame_moto_hex_list = response.get('frame_moto_hex_list', [])

            # Convert frame_anotado back to image
            frame_anotado_bytes = bytes.fromhex(frame_anotado_hex)
            nparr = np.frombuffer(frame_anotado_bytes, np.uint8)
            frame_anotado = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Convert frame_moto_hex_list back to images
            frame_moto_list = []
            for frame_moto_hex in frame_moto_hex_list:
                if frame_moto_hex:
                    frame_moto_bytes = bytes.fromhex(frame_moto_hex)
                    nparr = np.frombuffer(frame_moto_bytes, np.uint8)
                    frame_moto = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    frame_moto_list.append(frame_moto)
                else:
                    frame_moto_list.append(None)

            # Add to receive queue
            await receive_queue.put((frame_id, frame_anotado, detected_moto, detected_semaforo, frame_moto_list, infracoes_detectadas))
        except websockets.exceptions.ConnectionClosed:
            break
        except Exception as e:
            print(f"Error receiving frames: {e}")
            break

async def process_frames_task(receive_queue, window):
    while True:
        if not receive_queue.empty():
            frame_id, frame_anotado, detected_moto, detected_semaforo, frame_moto_list, infracoes_detectadas = await receive_queue.get()
            # Emit signals to update the GUI
            window.update_frame_signal.emit(frame_anotado)
            window.update_infractions_signal.emit(frame_moto_list, infracoes_detectadas)
        else:
            await asyncio.sleep(0.01)

if __name__ == '__main__':
    uri = "ws://192.168.1.5:8000/ws"
    video_path = 0  # Use 0 for webcam or provide video path
    video_path = 'videos/video_30s.mp4'

    app = QApplication(sys.argv)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    window = MainWindow()
    window.show()

    with loop:
        loop.create_task(send_frames(uri, video_path, window))
        loop.run_forever()
