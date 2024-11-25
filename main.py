import cv2

from ultralytics import YOLO
from config import (INPUT_VIDEO_PATH, OUTPUT_FOLDER, INFRACTIONS_FOLDER,
                    YOLO_MODEL_PATH)
from utils import (criar_pasta_saida, obter_caminho_saida, obter_propriedades_video)
from object_tracking import inicializar_rastreadores
from video_processing import processar_frame

def processar_video(arquivo_entrada, arquivo_saida):
    """Processa o vídeo e salva o resultado anotado."""
    cap = cv2.VideoCapture(arquivo_entrada)
    largura, altura, fps = obter_propriedades_video(cap)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
    out = cv2.VideoWriter(arquivo_saida, fourcc, fps, (largura, altura))

    # Inicializar rastreadores DeepSORT
    tracker_motos, tracker_semaforos = inicializar_rastreadores()
    trackers = (tracker_motos, tracker_semaforos)

    estado_semaforo = ['NOT RED']  # Estado atual do semáforo
    posicao_linha_parada = [None]  # Posição da linha de parada
    infracoes_detectadas = []
    frame_count = 0
    semaforo_mais_alto = {'dados': None}

    # Carrega o modelo YOLO
    modelo = YOLO(YOLO_MODEL_PATH)

    # Inicializar contadores de detecção e placas não reconhecidas
    last_moto_detection_frame = 0
    last_semaforo_detection_frame = 0
    unrecognized_plate_counter = [1]  # Usando lista para mutabilidade

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_original = frame.copy()

        frame_anotado, detected_moto, detected_semaforo = processar_frame(
            frame, modelo, trackers, estado_semaforo, posicao_linha_parada,
            infracoes_detectadas, frame_count, fps, semaforo_mais_alto,
            frame_original, arquivo_entrada,
            unrecognized_plate_counter
        )
        out.write(frame_anotado)

        # Atualizar últimos frames de detecção
        if detected_moto:
            last_moto_detection_frame = frame_count
        if detected_semaforo:
            last_semaforo_detection_frame = frame_count

        # Verificar se é necessário resetar os rastreadores
        if frame_count - last_moto_detection_frame > 30 * fps:
            # Resetar rastreador de motocicletas
            tracker_motos, _ = inicializar_rastreadores()
            trackers = (tracker_motos, tracker_semaforos)
            print('Resetando rastreador de motocicletas')
            last_moto_detection_frame = frame_count  # Atualizar para evitar múltiplos resets

        if frame_count - last_semaforo_detection_frame > 30 * fps:
            # Resetar rastreador de semáforos
            _, tracker_semaforos = inicializar_rastreadores()
            trackers = (tracker_motos, tracker_semaforos)
            print('Resetando rastreador de semáforos')
            last_semaforo_detection_frame = frame_count  # Atualizar para evitar múltiplos resets

        frame_count += 1

        # Opcional: Mostrar o frame em uma janela
        # cv2.imshow('Detecção de Infrações', frame_anotado)
        # if cv2.waitKey(1) == 27:
        #     break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    arquivo_entrada = INPUT_VIDEO_PATH

    criar_pasta_saida(OUTPUT_FOLDER)
    criar_pasta_saida(INFRACTIONS_FOLDER)
    arquivo_saida = obter_caminho_saida(arquivo_entrada, OUTPUT_FOLDER)

    processar_video(arquivo_entrada, arquivo_saida)
