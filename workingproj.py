# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 14:05:36 2020

@author: LENOVO:Mufaddal Hatim Hamid
"""
import numpy as np
import cv2
from keras.models import load_model
model=load_model('cat_dog_100epochs.h5')
roi1=(443, 114, 194, 188)
background = None

# Start with a halfway point between 0 and 1 of accumulated weight
accumulated_weight = 0.5
def calc_accum_avg(frame, accumulated_weight):
    '''
    Given a frame and a previous accumulated weight, computed the weighted average of the image passed in.
    '''
    
    # Grab the background
    global background
    
    # For first time, create the background from a copy of the frame.
    if background is None:
        background = frame.copy().astype("float")
        return None

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(frame, background, accumulated_weight)
def segment(frame, threshold=25):
    global background
    
    # Calculates the Absolute Differentce between the backgroud and the passed in frame
    diff = cv2.absdiff(background.astype("uint8"), frame)

    # Apply a threshold to the image so we can grab the foreground
    # We only need the threshold, so we will throw away the first item in the tuple with an underscore _
    _ , thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    return thresholded

def detection(frame1):
    
    frame1=cv2.resize(frame1,(150,150))
    frame1 = np.expand_dims(frame1, axis=0)
    frame1 = frame1/255
    #print(np.shape(frame1))
    prediction=model.predict_classes(frame1)
    predictpercent=model.predict(frame1)
    #print(str(predictpercent)[4:6])
    answer=str(prediction)
    answer=answer[2]
    return answer,int(str(predictpercent)[4:6])
    

# Read video
cap = cv2.VideoCapture(0)

# Read first frame.
ret, frame = cap.read()

# ROI DRAWN AT A FIXED CAMERA POSITION TO INCREASE ACCURACY
roi = roi1
num_frames=0
while True:
    # Read a new frame
    ret, frame = cap.read()
    frame=cv2.flip(frame,1)
    frame_copy=frame.copy()

       # Apply grayscale and blur to ROI
    (x,y,w,h) = tuple(map(int,roi))
    frame_copy=frame[y:y+h,x:x+h]
    gray = cv2.GaussianBlur(frame_copy, (3,3), 0)
   #SEPERATING BACKGROUND FROM FOREGROUND INCREASES DETECTION ACCURACY
    if num_frames < 60:
        calc_accum_avg(gray, accumulated_weight)
        if num_frames <= 59:
            cv2.putText(frame_copy, "WAIT! GETTING BACKGROUND AVG.", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

     #CHECK IF IMAGE IT CAT/DOG/NEITHER OF BOTH
    else:
        thresholded1=segment(gray)
        p1 = (x, y)
        p2 = (x+w, y+h)
        cv2.rectangle(frame, p1, p2, (255,0,0), 3)
        accept,value=detection(gray)
        if(value<25 and accept=="0"):
            cv2.putText(frame, "NO CAT OR DOG DETECTED", (100,200), cv2.FONT_HERSHEY_SIMPLEX, 1,(10,247,255),3)
        elif(accept=="0" and value>=25):
            cv2.putText(frame, "CAT!!", (100,200), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)    
        elif(accept=="1" and value>=55):
            cv2.putText(frame, "DOG", (100,200), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
        cv2.imshow("page",gray)
        cv2.imshow("Thesholded", thresholded1)
        cv2.imshow("working", frame)    
    
    num_frames+=1

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : 
        break
        
cap.release()
cv2.destroyAllWindows()
