import cv2
import numpy as np
import pytesseract
from utils import calcular_linha_parada, formatar_data_hora
from object_tracking import atualizar_rastreadores
from infractions import salvar_infracao
from utils import adicionar_texto_utf8
from config import (CLASSES_OF_INTEREST, CONFIDENCE_THRESHOLD, SEMAFORO_ALTURA_REAL,
                    DISTANCIA_SEMAFORO_LINHA_PARADA, TESSERACT_CMD)

def aplicar_desfoque_placas(frame, det_placas):
    """Aplica desfoque nas placas detectadas no frame."""
    for det in det_placas:
        x1p, y1p, wp, hp = det[0]
        x2p, y2p = x1p + wp, y1p + hp

        # Garantir que as coordenadas estejam dentro dos limites do frame
        altura_frame, largura_frame, _ = frame.shape
        x1p = max(0, x1p)
        y1p = max(0, y1p)
        x2p = min(largura_frame, x2p)
        y2p = min(altura_frame, y2p)

        # Aplicar desfoque na região da placa
        roi = frame[y1p:y2p, x1p:x2p]
        if roi.size == 0:
            continue
        roi_blur = cv2.GaussianBlur(roi, (51, 51), 0)
        frame[y1p:y2p, x1p:x2p] = roi_blur

def processar_frame(
        frame, modelo, trackers, estado_semaforo, posicao_linha_parada,
        infracoes_detectadas, frame_count, fps, semaforo_mais_alto, frame_original,
        arquivo_entrada, unrecognized_plate_counter
    ):
    """Processa um único frame e retorna o frame anotado."""
    altura_frame, largura_frame, _ = frame.shape

    resultados = modelo(frame)

    # Mapear nomes de classes para IDs numéricos
    class_name_to_id = {'motorcycle': 1, 'traffic light': 2, 'license plate': 3}

    detections = []

    detected_moto = False
    detected_semaforo = False

    for resultado in resultados:
        boxes = resultado.boxes
        for box in boxes:
            cls = int(box.cls[0])
            confianca = float(box.conf[0])
            rotulo = modelo.names[cls]

            if rotulo in CLASSES_OF_INTEREST and confianca > CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width = x2 - x1
                height = y2 - y1
                # Formato esperado: [x1, y1, width, height], confianca, class_id
                detection = [
                    [x1, y1, width, height],
                    confianca,
                    class_name_to_id[rotulo]
                ]
                detections.append(detection)

    # Separar detecções de motocicletas, semáforos e placas
    det_motos = [det for det in detections if det[2] == class_name_to_id['motorcycle']]
    det_semaforos = [det for det in detections if det[2] == class_name_to_id['traffic light']]
    det_placas = [det for det in detections if det[2] == class_name_to_id['license plate']]

    if det_motos:
        detected_moto = True
    if det_semaforos:
        detected_semaforo = True

    tracker_motos, tracker_semaforos = trackers

    # Atualizar rastreadores
    tracks_motos, tracks_semaforos = atualizar_rastreadores(
        tracker_motos, tracker_semaforos, det_motos, det_semaforos, frame
    )

    # Processar semáforos
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
        # Processar o estado do semáforo
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Detectar cor vermelha
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

    # Selecionar o semáforo mais alto no frame
    if semaforos:
        semaforo_mais_alto_atual = min(semaforos, key=lambda s: s['posicao'][1])
        # Atualizar o semáforo mais alto
        semaforo_mais_alto['dados'] = semaforo_mais_alto_atual
        estado_semaforo[0] = semaforo_mais_alto_atual['estado']
    else:
        semaforo_mais_alto['dados'] = None
        estado_semaforo[0] = 'NOT RED'

    # Calcular a posição da linha de parada
    if semaforo_mais_alto['dados'] is not None:
        posicao_semaforo = semaforo_mais_alto['dados']['posicao']
        tamanho_semaforo_pixels = semaforo_mais_alto['dados']['tamanho_pixels']
        posicao_linha_parada[0] = calcular_linha_parada(
            posicao_semaforo, tamanho_semaforo_pixels, altura_frame
        )
    else:
        # Se não for possível detectar o semáforo, usar posição padrão
        if posicao_linha_parada[0] is None:
            posicao_linha_parada[0] = int(altura_frame * 0.75)  # Ajuste conforme necessário

    # Desenhar a linha de parada
    cv2.line(frame, (0, posicao_linha_parada[0]),
             (largura_frame, posicao_linha_parada[0]), (255, 0, 0), 2)
    cv2.putText(frame, f'Sinal: {estado_semaforo[0]}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Configurar o Tesseract OCR, se necessário
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    # Processar motocicletas
    for track in tracks_motos:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        bbox = track.to_ltwh()
        x1, y1, w, h = map(int, bbox)
        x2, y2 = x1 + w, y1 + h
        centroide = ((x1 + x2) // 2, (y1 + y2) // 2)

        # **Adicionar verificação para garantir que posicao_linha_parada[0] não seja None**
        if posicao_linha_parada[0] is not None:
            passou_linha = centroide[1] < posicao_linha_parada[0]
        else:
            # Se posicao_linha_parada[0] for None, não podemos determinar se passou a linha
            passou_linha = False

        # Verificar se a motocicleta já cometeu infração
        infracao_existente = next((inf for inf in infracoes_detectadas if inf['id_moto'] == track_id), None)

        if passou_linha and estado_semaforo[0] == 'RED':
            if infracao_existente is None:
                # Motocicleta cruzou a linha de parada com sinal vermelho
                # Extrair placa (se disponível)
                placa_texto = 'placa_nao_identificada'
                # Buscar placas dentro da bounding box da moto
                placa_encontrada = False
                for det in det_placas:
                    x1p, y1p, wp, hp = det[0]
                    x2p, y2p = x1p + wp, y1p + hp
                    # Verificar se a placa está dentro da moto
                    if x1p >= x1 and y1p >= y1 and x2p <= x2 and y2p <= y2:
                        placa_roi = frame_original[y1p:y2p, x1p:x2p]
                        # Realizar OCR na placa
                        placa_texto_ocr = pytesseract.image_to_string(
                            placa_roi, config='--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                        )
                        placa_texto_ocr = placa_texto_ocr.strip().replace(' ', '')
                        if placa_texto_ocr:
                            placa_texto = placa_texto_ocr
                        else:
                            # Placa não reconhecida
                            placa_texto = f'placa_nao_reconhecida_{unrecognized_plate_counter[0]}'
                            unrecognized_plate_counter[0] += 1
                        placa_encontrada = True
                        break
                if not placa_encontrada:
                    # Placa não detectada dentro da moto
                    placa_texto = f'placa_nao_reconhecida_{unrecognized_plate_counter[0]}'
                    unrecognized_plate_counter[0] += 1

                # Salvar imagem da motocicleta
                moto_roi = frame_original[y1:y2, x1:x2]
                # Salvar infração em pasta separada
                salvar_infracao(
                    moto_roi,
                    placa_texto,
                    frame_original,
                    (x1, y1, x2, y2),
                    infracoes_detectadas,
                    track_id,
                    semaforo_mais_alto,
                    posicao_linha_parada[0],
                    arquivo_entrada,
                    frame_count,  # Passe o frame atual
                    fps
                )
                # Desenhar alerta
                adicionar_texto_utf8(
                    frame,
                    f'INFRAÇÃO Moto {track_id}',
                    (x1 + 20, y1 - 20),
                    (0, 0, 255)
                )
            else:
                # Já foi registrada a infração
                pass

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(frame, f'Moto {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Aplicar desfoque nas placas detectadas (Fazer isso por último, após todos os processos)
    aplicar_desfoque_placas(frame, det_placas)

    # Adicionar data e hora no canto inferior direito
    data_hora = formatar_data_hora()
    cv2.putText(frame, data_hora, (largura_frame - 300, altura_frame - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame, detected_moto, detected_semaforo
