import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

path = 'dataset'
images = []
classNames = []

# Load images
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# Encode faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Mark attendance (fixed version)
def markAttendance(name):
    now = datetime.now()
    dateString = now.strftime('%Y-%m-%d')
    timeString = now.strftime('%H:%M:%S')

    with open('attendance.csv', 'a') as f:
        f.writelines(f'\n{name},{dateString},{timeString}')

# Prepare encodings
encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

marked_names = set()
last_name = ""   # 🔥 for stability

while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):

        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

        # 🔥 Recognition with stability + UNKNOWN
        if matches[matchIndex] and faceDis[matchIndex] < 0.5:
            name = classNames[matchIndex].upper()
            last_name = name

            if name not in marked_names:
                markAttendance(name)
                marked_names.add(name)
        else:
            name = last_name if last_name != "" else "UNKNOWN"

        # Draw box
        y1,x2,y2,x1 = faceLoc
        y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4

        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
        cv2.putText(img,name,(x1+6,y2-6),
                    cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

    cv2.imshow('Face Attendance System', img)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
