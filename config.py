import os

# Caminhos dos arquivos
YOLO_MODEL_PATH = 'yolo/yolo11x.pt'

# INPUT_VIDEO_PATH = 'videos/porto_alegre.mp4'
# INPUT_VIDEO_PATH = 'videos/2024-11-12 14-14-33.mp4'
INPUT_VIDEO_PATH = 'videos/dirigindo_pela_primeira_vez_no_centro_da_cidade!_Av_Afonso_Pena!.mp4'
# INPUT_VIDEO_PATH = 0
# INPUT_VIDEO_PATH = 'videos/2024-11-12 14-26-04.mp4'

OUTPUT_FOLDER = 'videos_output'
INFRACTIONS_FOLDER = 'infractions'

# Classes de interesse
CLASSES_OF_INTEREST = ['motorcycle', 'traffic light', 'license plate']

# Parâmetros do DeepSORT
DEEPSORT_MAX_AGE = 30

# Parâmetros de detecção
CONFIDENCE_THRESHOLD = 0.3

# Parâmetros físicos
SEMAFORO_ALTURA_REAL = 0.6  # 0,2*3 metros: Semáforo, Sinaleira ou Farol de 3 
# Focais de 200mm, possui às características Técnicas Gerais que  atende as normas NBR 7995 da ABNT.
# https://www.ecosemaforo.com.br/ecosemaforos-e-semaforos/semaforo-tipo-i.html
DISTANCIA_SEMAFORO_LINHA_PARADA = 5.5  # metros: A altura livre do anteparo dos grupos focais 
# em projeção sobre a via deve ser de 5,50 metros,
# podendo variar em situações específicas (ver Figura 10.4). 
# pag 207  https://www.gov.br/transportes/pt-br/assuntos/transito/arquivos-senatran/docs/copy_of___05___MBST_Vol._V___Sinalizacao_Semaforica.pdf


# Parâmetros de salvamento de vídeo
DURACAO_PRE_INFRACTION = 15  # segundos
DURACAO_POS_INFRACTION = 15  # segundos

# Configuração do Tesseract OCR (se necessário)
# TESSERACT_CMD = r'/usr/bin/tesseract'
TESSERACT_CMD = None  # Defina o caminho se necessário

# Certifique-se de que as pastas de saída existem
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(INFRACTIONS_FOLDER, exist_ok=True)
