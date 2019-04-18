"""
@author: QArif

Purpose of file is to test camera
"""
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# import the good stuff 
import numpy as np
import cv2
import random
from utils import CFEVideoConf, image_resize
import glob
import math

# define
cap = cv2.VideoCapture(0)
# frames_per_seconds = 20
# save_path='saved-media/filter.mp4'
# out = cv2.VideoWriter(save_path, config.video_type, frames_per_seconds, config.dims)

def alpha_blend(frame_1, frame_2, mask):
    alpha = mask/255.0 
    blended = cv2.convertScaleAbs(frame_1*(1-alpha) + frame_2*alpha)
    return blended

def apply_circle_focus_blur(frame, intensity=0.2):
    frame_h, frame_w, frame_c = frame.shape
    y = int(frame_h/2)
    x = int(frame_w/2)

    mask = np.zeros((frame_h, frame_w, 4), dtype='uint8')
    cv2.circle(mask, (x, y), int(y/2), (255,255,255), -1, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (21,21),11 )

    blured = cv2.GaussianBlur(frame, (21,21), 11)
    blended = alpha_blend(frame, blured, 255-mask)
    frame = cv2.cvtColor(blended, cv2.COLOR_BGRA2BGR)
    return frame

def portrait_mode(frame):
#    cv2.imshow('frame', frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 120,255,cv2.THRESH_BINARY)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
    blured = cv2.GaussianBlur(frame, (21,21), 11)
    blended = alpha_blend(frame, blured, mask)
    frame = cv2.cvtColor(blended, cv2.COLOR_BGRA2BGR)
    return frame

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    #Verify alpha channel
    try:
        frame.shape[3] # looking for the alpha channel
    except IndexError:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    # Filter frame
    blur_mask = apply_circle_focus_blur(frame.copy())
    portrait = portrait_mode(frame.copy())
   
    # Display the resulting frame
    cv2.imshow('frame',frame)
    # cv2.imshow('portrait',portrait)
    cv2.imshow('blur_mask', blur_mask)    

    
    # Break loop by pressing 'q'. Press to exit
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# function to set resolution
# def make_1080p():
#        cap.set(3, 1920)
#        cap,set(4, 1080)