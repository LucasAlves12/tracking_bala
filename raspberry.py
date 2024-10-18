import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

# Inicializa a câmera Pi
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(640, 480))

# Permite que a câmera aqueça
time.sleep(0.1)

# Inicializa o detector de objetos
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

# Loop principal para capturar e processar os frames
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array

    # Aplica filtro Gaussiano para suavizar a imagem e reduzir ruído
    blurred_frame = cv2.GaussianBlur(image, (5, 5), 0)

    # Aplica o detector de objetos
    mask = object_detector.apply(blurred_frame)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

    # Operações morfológicas para remover ruído
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Erosão e dilatação para refinar a máscara
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Encontra contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # Filtra pequenas áreas
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Mostra os frames processados
    cv2.imshow("Frame", image)
    cv2.imshow("Mask", mask)

    # Limpa o stream para o próximo frame
    rawCapture.truncate(0)

    # Verifica se a tecla 'q' foi pressionada para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
cv2.destroyAllWindows()
