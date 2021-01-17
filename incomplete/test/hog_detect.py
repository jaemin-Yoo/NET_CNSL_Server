import os
import cv2


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture('test_cut.mp4')

while True:
    ret, frame_rotate = cap.read()
    frame = cv2.transpose(frame_rotate)
    frame = cv2.flip(frame, 1)

    if not ret:
        break

    detected, _ = hog.detectMultiScale(frame)

    for (x, y, w, h) in detected:
        cv2.rectangle(frame, (x, y, w, h), (0, 255, 0), 3)

    cv2.imshow('Detect', frame)

    if cv2.waitKey(10) == 27:
        break

cv2.destroyAllWindows()
