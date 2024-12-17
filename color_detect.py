import cv2
import numpy as np
import random
''' жёлтый
lower_red = np.array([20, 100, 0])
upper_red = np.array(([80, 255, 255]))'''
''' синий
lower_red = np.array([100, 0, 0])
upper_red = np.array(([300, 255, 255]))'''
''' зелёный
lower_red = np.array([40, 100, 100])
upper_red = np.array(([80, 255, 255]))'''
image = cv2.imread('image.jpg')
lower_red = np.array([0, 100, 160])
upper_red = np.array(([100, 255, 255]))
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        print('Ошибка: Не удалось получить кадр')
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(hsv, lower_red, upper_red)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    for (x, y, w,h) in faces:
        #  cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.line(frame, (x+20, y+30), (x+random.randint(0,255) + w+25, y+200 + h+25), (0, 0, 255), 15)
        cv2.rectangle(image, (x, y), (x + w, y + h), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 100)




    faces = face_cascade.detectMultiScale(mask, scaleFactor=1.1, minNeighbors=5)


    cv2.imshow('original', frame)
    #  cv2.imshow('mask', mask)
    #  cv2.imshow('result', result)
    #  cv2.imshow('image', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()