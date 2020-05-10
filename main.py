# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 15:37:11 2017

@author: pc
"""

import cv2
import numpy as np
import two_players
import one_player
import time
import forExp2

v=0

camera = cv2.VideoCapture(0)
tzz = time.time()
while(1):
    # Capture frame from camera
    ret, frame = camera.read()
    frame=cv2.flip(frame,1)
    frame_original = np.copy(frame)
    
    
    
    cv2.putText(frame,"Choose mode for game:",(int(0.40*frame.shape[1]),int(0.27*frame.shape[0])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,8)
    cv2.putText(frame,"Show index finger for single player mode",(int(0.28*frame.shape[1]),int(0.47*frame.shape[0])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,8) 
    cv2.putText(frame,"Show thumb and little finger for double player mode",(int(0.28*frame.shape[1]),int(0.67*frame.shape[0])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,8) 
    #cv2.putText(frame,"Show thumbs up for rules",(int(0.28*frame.shape[1]),int(0.47*frame.shape[0])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1,8)
    
    if(time.time()-tzz>=5):
        x = forExp2.recognise(frame_original, 0)
    
        if(x == 1):
            camera.release()
            cv2.destroyAllWindows()
            v = 1
            break
        
        elif(x == 2):
            camera.release()
            cv2.destroyAllWindows()
            v = 2
            break
        
    cv2.imshow("Welcome to Hand Gesture RPSLS", frame)
    
    interrupt=cv2.waitKey(10)
    # Quit by pressing 'q'
    if  interrupt & 0xFF == ord('q'):
        camera.release()
        cv2.destroyAllWindows()
        break
        
if(v==1):
    one_player.one_player()
elif(v==2):
    two_players.two_players()
    
camera.release()
cv2.destroyAllWindows()