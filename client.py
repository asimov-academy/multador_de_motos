import cv2
import asyncio
import websockets
import json
import numpy as np

async def send_frames(uri, video_path):
    async with websockets.connect(uri) as websocket:
        cap = cv2.VideoCapture(video_path)  # Ou use 0 para webcam
        frame_id = 0
        running = True

        # Filas para gerenciar envio e recebimento
        send_queue = asyncio.Queue()
        receive_queue = asyncio.Queue()

        # Iniciar tarefas de envio e recebimento
        send_task = asyncio.create_task(send_frames_task(cap, websocket, send_queue, receive_queue, frame_id, running))
        receive_task = asyncio.create_task(receive_frames_task(websocket, receive_queue, running))
        process_task = asyncio.create_task(process_frames_task(receive_queue))

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

        # Codificar frame em bytes e converter para hexadecimal
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        frame_hex = frame_bytes.hex()

        # Preparar a mensagem a ser enviada
        message = {
            'frame_id': frame_id,
            'frame': frame_hex,
            # Outros parâmetros, se necessário
        }

        # Enviar mensagem para o servidor
        await websocket.send(json.dumps(message))

        # Opcionalmente, armazenar o frame original associado ao frame_id
        # await send_queue.put((frame_id, frame))

        # Controlar a taxa de envio, se necessário
        await asyncio.sleep(0)  # Pode ajustar o valor conforme necessário

async def receive_frames_task(websocket, receive_queue, running):
    while running:
        try:
            response_data = await websocket.recv()
            response = json.loads(response_data)

            frame_id = response['frame_id']
            frame_anotado_hex = response['frame_anotado']
            detected_moto = response['detected_moto']
            detected_semaforo = response['detected_semaforo']

            # Converter frame anotado de volta para imagem
            frame_anotado_bytes = bytes.fromhex(frame_anotado_hex)
            nparr = np.frombuffer(frame_anotado_bytes, np.uint8)
            frame_anotado = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Adicionar à fila de recebimento
            await receive_queue.put((frame_id, frame_anotado, detected_moto, detected_semaforo))
        except websockets.exceptions.ConnectionClosed:
            break

async def process_frames_task(receive_queue):
    while True:
        if not receive_queue.empty():
            frame_id, frame_anotado, detected_moto, detected_semaforo = await receive_queue.get()

            # Exibir frame anotado
            cv2.imshow('Frame Anotado', frame_anotado)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            await asyncio.sleep(0.01)  # Pequeno delay para evitar loop ocupado

if __name__ == '__main__':
    uri = "ws://192.168.1.5:8000/ws"
    video_path = 'videos/2024-11-12 14-14-33.mp4'
    video_path = 'videos/dirigindo_pela_primeira_vez_no_centro_da_cidade!_Av_Afonso_Pena!.mp4'
    video_path = 0

    asyncio.get_event_loop().run_until_complete(send_frames(uri, video_path))