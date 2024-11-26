import os
import cv2
from datetime import datetime
from utils import adicionar_texto_utf8
from config import DURACAO_PRE_INFRACTION, DURACAO_POS_INFRACTION, INFRACTIONS_FOLDER

def salvar_infracao(
    moto_roi, placa_texto, frame, bbox_moto, infracoes_detectadas,
    track_id, semaforo_mais_alto, linha_parada_y, arquivo_entrada,
    tempo_infracao_frame, fps
):
    """Salva todos os arquivos relacionados à infração em uma pasta separada."""
    data_hora = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # Usar o número da placa no nome da pasta
    nome_pasta = f'{placa_texto}_{data_hora}'
    caminho_pasta = os.path.join(INFRACTIONS_FOLDER, nome_pasta)
    os.makedirs(caminho_pasta, exist_ok=True)

    # Salvar imagem da motocicleta
    caminho_moto = os.path.join(caminho_pasta, f'motocicleta_{track_id}.jpg')
    cv2.imwrite(caminho_moto, moto_roi)

    # Salvar frame com anotação
    frame_anotado = frame.copy()

    x1, y1, x2, y2 = bbox_moto
    cv2.rectangle(frame_anotado, (x1, y1), (x2, y2), (0, 0, 255), 2)

    frame_anotado = adicionar_texto_utf8(
        frame_anotado,
        f'INFRAÇÃO Moto {track_id}',
        (x1 + 20, y1 - 20),
        (0, 0, 255)
    )

    if semaforo_mais_alto['dados'] and semaforo_mais_alto['dados']['bbox']:
        x1_s, y1_s, x2_s, y2_s = semaforo_mais_alto['dados']['bbox']
        cv2.rectangle(frame_anotado, (x1_s, y1_s), (x2_s, y2_s), (0, 0, 255), 2)

        frame_anotado = adicionar_texto_utf8(
            frame_anotado,
            f'Semáforo {semaforo_mais_alto["dados"]["id"]} (RED)',
            (x1_s + 20, y1_s - 10),
            (0, 0, 255)
        )

    # Desenhar linha de parada
    altura_frame, largura_frame, _ = frame_anotado.shape
    cv2.line(frame_anotado, (0, linha_parada_y),
             (largura_frame, linha_parada_y), (255, 0, 0), 2)
    cv2.putText(frame_anotado, 'Linha de Parada', (50, linha_parada_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    caminho_frame_anotado = os.path.join(caminho_pasta, f'frame_infracao_{track_id}.jpg')
    cv2.imwrite(caminho_frame_anotado, frame_anotado)

    # Salvar vídeo original e vídeo com anotações
    salvar_videos_infracao(
        arquivo_entrada,
        caminho_pasta,
        tempo_infracao_frame,
        fps,
        track_id,
        semaforo_mais_alto,
        linha_parada_y,
        bbox_moto
    )

    # Adicionar à lista de infrações
    infracoes_detectadas.append({
        'id_moto': track_id,
        'tempo_infracao': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
        'placa': placa_texto,
        'pasta': caminho_pasta,
        'tempo_infracao_frame': tempo_infracao_frame  # Adicione esta linha
    })

def salvar_videos_infracao(
    arquivo_entrada, caminho_pasta, tempo_infracao_frame, fps, track_id,
    semaforo_mais_alto, linha_parada_y, bbox_moto,
    duracao_pre=DURACAO_PRE_INFRACTION, duracao_pos=DURACAO_POS_INFRACTION
):
    """Salva o trecho do vídeo original e o vídeo com anotações."""
    cap = cv2.VideoCapture(arquivo_entrada)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    inicio_frame = max(0, tempo_infracao_frame - int(duracao_pre * fps))
    fim_frame = min(total_frames, tempo_infracao_frame + int(duracao_pos * fps))

    # Vídeo original
    caminho_video_original = os.path.join(caminho_pasta, f'video_original_{track_id}.mp4')
    # Vídeo com anotações
    caminho_video_anotado = os.path.join(caminho_pasta, f'video_anotado_{track_id}.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_original = cv2.VideoWriter(caminho_video_original, fourcc, fps, (largura, altura))
    out_anotado = cv2.VideoWriter(caminho_video_anotado, fourcc, fps, (largura, altura))

    cap.set(cv2.CAP_PROP_POS_FRAMES, inicio_frame)
    frame_idx = inicio_frame

    while frame_idx < fim_frame:
        ret, frame = cap.read()
        if not ret:
            break

        frame_anotado = frame.copy()

        # Adicionar data e hora no canto inferior direito
        data_hora_frame = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        cv2.putText(frame_anotado, data_hora_frame, (largura - 300, altura - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Adicionar anotações
        # Desenhar motocicleta
        x1, y1, x2, y2 = bbox_moto
        cv2.rectangle(frame_anotado, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame_anotado, f'Moto {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        # Desenhar semáforo
        if semaforo_mais_alto['dados'] and semaforo_mais_alto['dados']['bbox']:
            x1_s, y1_s, x2_s, y2_s = semaforo_mais_alto['dados']['bbox']
            cv2.rectangle(frame_anotado, (x1_s, y1_s), (x2_s, y2_s), (0, 0, 255), 2)

            frame_anotado = adicionar_texto_utf8(
                frame_anotado,
                f'Semáforo {semaforo_mais_alto["dados"]["id"]} (RED)',
                (x1_s + 20, y1_s - 10),
                (0, 0, 255)
            )

        # Desenhar linha de parada
        cv2.line(frame_anotado, (0, linha_parada_y),
                 (largura, linha_parada_y), (255, 0, 0), 2)
        cv2.putText(frame_anotado, 'Linha de Parada', (50, linha_parada_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        out_original.write(frame)
        out_anotado.write(frame_anotado)
        frame_idx += 1

    cap.release()
    out_original.release()
    out_anotado.release()
