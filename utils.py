import os
import cv2
import numpy as np

from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from config import (SEMAFORO_ALTURA_REAL, DISTANCIA_SEMAFORO_LINHA_PARADA)

def criar_pasta_saida(nome_pasta):
    """Cria a pasta de saída se ela não existir."""
    if not os.path.exists(nome_pasta):
        os.makedirs(nome_pasta)
    return nome_pasta

def obter_caminho_saida(arquivo_entrada, pasta_saida):
    """Gera o caminho do arquivo de saída com base no arquivo de entrada."""
    nome_base = os.path.basename(arquivo_entrada)
    nome, extensao = os.path.splitext(nome_base)
    arquivo_saida = os.path.join(pasta_saida, f'{nome}_output{extensao}')
    return arquivo_saida

def obter_propriedades_video(cap):
    """Obtém as propriedades do vídeo."""
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None or np.isnan(fps):
        fps = 20  # Valor padrão se o FPS não puder ser obtido
    return largura, altura, fps

def calcular_linha_parada(posicao_semaforo, tamanho_semaforo_pixels, altura_frame):
    """
    Calcula a posição da linha de parada com base na posição e tamanho do semáforo.

    Assume-se que o semáforo tem dimensões físicas conhecidas, e que a altura do
    semáforo em pixels na imagem pode ser usada para calcular a escala de pixels
    para metros. Com essa escala, podemos estimar a distância em pixels entre o
    semáforo e a linha de parada.

    Parâmetros:
    - posicao_semaforo: tupla (x, y) representando o centro do semáforo detectado.
    - tamanho_semaforo_pixels: altura do semáforo em pixels.
    - altura_frame: altura do frame em pixels.
    - altura_semaforo_real: altura real do semáforo em metros.
    - distancia_semaforo_linha_parada: distância real do semáforo à linha de parada em metros.

    Retorna:
    - y_linha_parada: posição Y estimada da linha de parada no frame.
    """

    # Calcula a escala de pixels por metro
    escala_pixels_por_metro = tamanho_semaforo_pixels / SEMAFORO_ALTURA_REAL

    # Calcula a distância em pixels entre o semáforo e a linha de parada
    distancia_pixels = escala_pixels_por_metro * DISTANCIA_SEMAFORO_LINHA_PARADA

    # Calcula a posição Y da linha de parada
    y_semaforo = posicao_semaforo[1]
    y_linha_parada = int(y_semaforo + distancia_pixels)

    # Verifica se a linha de parada está dentro dos limites do frame
    y_linha_parada = min(y_linha_parada, altura_frame - 1)
    return y_linha_parada

def formatar_data_hora():
    """Retorna a data e hora atuais formatadas."""
    return datetime.now().strftime('%d/%m/%Y %H:%M:%S')

def adicionar_texto_utf8(imagem_cv, texto, posicao, cor):
    """Adiciona texto com suporte UTF-8 em uma imagem do OpenCV."""
    # Converte a imagem OpenCV para PIL
    imagem_pil = Image.fromarray(cv2.cvtColor(imagem_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(imagem_pil)
    
    # Localiza fonte no sistema
    fonte_ttf = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Caminho padrão no Linux
    if not os.path.exists(fonte_ttf):
        raise FileNotFoundError(f"Fonte não encontrada no caminho: {fonte_ttf}")
    
    fonte = ImageFont.truetype(fonte_ttf, 32)
    
    # Adiciona o texto na imagem
    draw.text(posicao, texto, font=fonte, fill=cor)
    
    # Converte de volta para OpenCV
    imagem_cv = cv2.cvtColor(np.array(imagem_pil), cv2.COLOR_RGB2BGR)
    return imagem_cv
