"""
@author: QArif

Purpose of file is to test camera
"""
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# import the good stuff 
import numpy as np
import cv2
import random
import glob
import math
import utils

# define
cap = cv2.VideoCapture(0)
# frames_per_seconds = 20
# save_path='saved-media/filter.mp4'
# out = cv2.VideoWriter(save_path, config.video_type, frames_per_seconds, config.dims)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Verify alpha channel
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