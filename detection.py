import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Haar trained model on frontal faces only
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while(True):
    ret, frame = cap.read() # read frame by frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert the image into gray scale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # draw a rectangle if the faces are detected, passed red in BGR format
    for (x,y,w,h) in faces:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (0 , 0, 255), 2)
    cv2.imshow('Output', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):#q terminates script
        break

# destroy all opened windows
cap.release()
cv2.destroyAllWindows()
