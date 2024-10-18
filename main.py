import cv2

# Inicializa a captura de vídeo da câmera do notebook
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Inicializa o detector de objetos
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Aplica filtro Gaussiano para suavizar a imagem e reduzir ruído
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

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
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Mostra os frames processados
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    # Verifica se a tecla 'q' foi pressionada para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()