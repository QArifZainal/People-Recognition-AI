# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 14:48:02 2019

@author: QArif

C:\ProgramData\Anaconda3\lib\site-packages\cv2\cv2.cp36-win_amd64.pyd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
"""

import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
# eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
# smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')
# recognizer = cv2.face.createEigenFaceRecognizer(15, 4000)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner.yml")

debug_mode = 0
labels = {"person_name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        if debug_mode ==1:
            print(x,y,w,h)
    roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
    roi_color = frame[y:y+h, x:x+w]
    id_, confidence = recognizer.predict(roi_gray)
    if confidence >=4 and confidence <= 85:
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = labels[id_]
        conf_value = str(round(confidence, 1))
        color = (255, 255, 255)
        stroke = 2
        cv2.putText(frame, "conf:", (x,y-64), font, 1, color, stroke, cv2.LINE_AA)
        cv2.putText(frame, name, (x,y-16), font, 1, color, stroke, cv2.LINE_AA)
        cv2.putText(frame, conf_value, (x+100,y-64), font, 1, color, stroke, cv2.LINE_AA)
        count = 0
    # take sample images
    while count < 3:
        img_item ="testframe" + str(count) + ".png"
        cv2.imwrite(img_item, frame)
        count += 1

    color = (255, 120, 120) #BGR 0-255
    stroke = 2
    end_cord_x = x + w
    end_cord_y = y + h
    cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
    	#subitems = smile_cascade.detectMultiScale(roi_gray)
    	#for (ex,ey,ew,eh) in subitems:

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
