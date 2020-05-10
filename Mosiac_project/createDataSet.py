# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 00:55:05 2017

@author: pc
"""

import cv2
import numpy as np
import math, time
import img_thresholding

capture_box_count=9
capture_box_dim=20
capture_box_sep_x=8
capture_box_sep_y=18
capture_pos_x=480
capture_pos_y=150
cap_region_x_begin=0.5 # start point/total width
cap_region_y_end=0.8 # start point/total width

   
camera = cv2.VideoCapture(0)
capture_done=0
#GestureDictionary=DefineGestures()
#frame_gesture=Gesture("frame_gesture")
j=0
while(1):
     # Capture frame from camera
     ret, frame = camera.read()
     frame=cv2.bilateralFilter(frame,5,50,100)
     # Operations on the frame
     frame=cv2.flip(frame,1)
     
     #cv2.rectangle(frame,(int(cap_region_x_begin*frame.shape[1]),0),(frame.shape[1],int(cap_region_y_end*frame.shape[0])),(255,0,0),1)
     frame_original=np.copy(frame)
     roi=np.copy(frame)
     
     if (not (capture_done)):
         cv2.putText(frame_original,"Place hand inside boxes and press 'c' to capture hand histogram",(int(0.08*frame.shape[1]),int(0.97*frame.shape[0])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1,8)
         first_iteration = True
         #box_pos_x=np.array([capture_pos_x,capture_pos_x+capture_box_dim+capture_box_sep_x,capture_pos_x+2*capture_box_dim+2*capture_box_sep_x,capture_pos_x+3*capture_box_dim+3*capture_box_sep_x,capture_pos_x,capture_pos_x+capture_box_dim+capture_box_sep_x,capture_pos_x+2*capture_box_dim+2*capture_box_sep_x,capture_pos_x+3*capture_box_dim+3*capture_box_sep_x,capture_pos_x,capture_pos_x+capture_box_dim+capture_box_sep_x,capture_pos_x+2*capture_box_dim+2*capture_box_sep_x,capture_pos_x+3*capture_box_dim+3*capture_box_sep_x,capture_pos_x,capture_pos_x+capture_box_dim+capture_box_sep_x,capture_pos_x+2*capture_box_dim+2*capture_box_sep_x,capture_pos_x+3*capture_box_dim+3*capture_box_sep_x,capture_pos_x,capture_pos_x+capture_box_dim+capture_box_sep_x,capture_pos_x+2*capture_box_dim+2*capture_box_sep_x,capture_pos_x+3*capture_box_dim+3*capture_box_sep_x],dtype=int)
         #box_pos_y=np.array([capture_pos_y,capture_pos_y,capture_pos_y,capture_pos_y,capture_pos_y+capture_box_dim+capture_box_sep_y,capture_pos_y+capture_box_dim+capture_box_sep_y,capture_pos_y+capture_box_dim+capture_box_sep_y,capture_pos_y+capture_box_dim+capture_box_sep_y,capture_pos_y+2*capture_box_dim+2*capture_box_sep_y,capture_pos_y+2*capture_box_dim+2*capture_box_sep_y,capture_pos_y+2*capture_box_dim+2*capture_box_sep_y,capture_pos_y+2*capture_box_dim+2*capture_box_sep_y,capture_pos_y+3*capture_box_dim+3*capture_box_sep_y,capture_pos_y+3*capture_box_dim+3*capture_box_sep_y,capture_pos_y+3*capture_box_dim+3*capture_box_sep_y,capture_pos_y+3*capture_box_dim+3*capture_box_sep_y,capture_pos_y+4*capture_box_dim+4*capture_box_sep_y,capture_pos_y+4*capture_box_dim+4*capture_box_sep_y,capture_pos_y+4*capture_box_dim+4*capture_box_sep_y,capture_pos_y+4*capture_box_dim+4*capture_box_sep_y],dtype=int)
         #box_pos_x=np.array([capture_pos_x,capture_pos_x+capture_box_dim+capture_box_sep_x,capture_pos_x+2*capture_box_dim+2*capture_box_sep_x,capture_pos_x,capture_pos_x+capture_box_dim+capture_box_sep_x,capture_pos_x+2*capture_box_dim+2*capture_box_sep_x,capture_pos_x,capture_pos_x+capture_box_dim+capture_box_sep_x,capture_pos_x+2*capture_box_dim+2*capture_box_sep_x],dtype=int)
         #box_pos_y=np.array([capture_pos_y,capture_pos_y,capture_pos_y,capture_pos_y+capture_box_dim+capture_box_sep_y,capture_pos_y+capture_box_dim+capture_box_sep_y,capture_pos_y+capture_box_dim+capture_box_sep_y,capture_pos_y+2*capture_box_dim+2*capture_box_sep_y,capture_pos_y+2*capture_box_dim+2*capture_box_sep_y,capture_pos_y+2*capture_box_dim+2*capture_box_sep_y],dtype=int)
    
         #for i in range(capture_box_count):
             #cv2.rectangle(frame_original,(box_pos_x[i],box_pos_y[i]),(box_pos_x[i]+capture_box_dim,box_pos_y[i]+capture_box_dim),(255,0,0),1)
     else:  
         k = str(0.125-(time.time()-t1))+str(j+1)
         #cv2.putText(frame_original,k,(int(0.08*frame.shape[1]),int(0.97*frame.shape[0])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1,8)
         roi = img_thresholding.roi(frame_original)
         #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
         #ret,roi1 = cv2.threshold(roi ,10,255,cv2.THRESH_BINARY)
         cv2.imshow("roi", roi)
         if(time.time()-t1>=0.125):
             j = j+1
             #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
             #ret,thresh1 = cv2.threshold(roi ,10,255,cv2.THRESH_BINARY)
             cv2.imwrite("4/"+str(j)+".jpg", roi)
             t1 = time.time()
             if(j==600):
                 break
             
     
     cv2.imshow('Hand Gesture Recognition v1.0',frame_original)
     
     interrupt=cv2.waitKey(10)
     # Quit by pressing 'q'
     if  interrupt & 0xFF == ord('q'):
         break
     # Capture hand by pressing 'c'
     elif interrupt & 0xFF == ord('c'):
         capture_done=1
         t1 = time.time()
     # Reset captured hand by pressing 'r'
     elif interrupt & 0xFF == ord('r'):
         capture_done=0