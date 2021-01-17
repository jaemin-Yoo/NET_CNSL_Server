import cv2

cascadefile = "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cascadefile)

print('Start')

def detect(gray, frame):
    faces = cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y, w, h), (0, 255, 0), 3)
        
    return frame

cap=cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    process=detect(gray, frame)
    cv2.imshow('Detect', process)
    
    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
