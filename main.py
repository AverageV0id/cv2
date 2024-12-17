import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Ошибка: Не удалось открыть камеру')
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print('Ошибка: Не удалось получить кадр')
        break

    cv2.imshow('camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()