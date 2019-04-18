# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:19:23 2019

@author: QArif
"""

# import the good stuff 
import numpy as np
import cv2
import pickle

# haarcascade 
face_cascade2 = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt_tree.xml')
fullbody_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_fullbody.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/train-set.yml")

labels = {"person_name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

# define
cap = cv2.VideoCapture(0)
# img = cv2.imread('fullbody.jpg')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
  
    # Verify alpha channel
    try:
        frame.shape[3] # looking for the alpha channel
    except IndexError:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
#    faces1 = face_cascade2.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    fullbody = fullbody_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    # Filter frame
    blur_mask = apply_circle_focus_blur(frame.copy())

    for (x, y, w, h) in faces:
    	#print(x,y,w,h)
        	roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
        	roi_color = frame[y:y+h, x:x+w]

    for (x, y, w, h) in fullbody:
        	roi_gray_body = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
        	roi_color_body = frame[y:y+h, x:x+w]

    	# recognize? deep learned model predict keras tensorflow pytorch scikit learn 
    id_, conf = recognizer.predict(roi_gray)

    if conf>=35 and conf <= 85:
    		print(labels[id_])
    		font = cv2.FONT_HERSHEY_SIMPLEX
    		name = labels[id_]
    		color = (255, 255, 255)
    		stroke = 2
    		cv2.putText(frame, name, (x,y-16), font, 1, color, stroke, cv2.LINE_AA)
    
    img_item = "capture_frame.png"
    cv2.imwrite(img_item, roi_color)
    
    color = (255, 120, 120) #BGR 0-255 
    stroke = 3
    end_cord_x = x + w
    end_cord_y = y + h
    cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)   
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.imshow('blur_mask', blur_mask)    

    # Break loop by pressing 'q'. Press to exit
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()