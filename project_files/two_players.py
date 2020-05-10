# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 17:27:53 2017

@author: pc
"""

import cv2
import numpy as np
import math, time
import forExp2
import rpsls

def number_to_name(n):
    
    predicted = n
    
    if predicted==0:
        return "Rock"
    elif predicted==1:
        return "Spock"
    elif predicted==2:
        return "Paper"
    elif predicted==3:
        return "Lizard"
    elif predicted==4:
        return "Scissor"


def two_players():
    c=0
    rounds = 0
    rounds1 = 1
    v_initial = 0
    w1 = 0
    w2 = 0
    start = 0
    www = 0
    start = 0
    tzz = time.time()
    camera = cv2.VideoCapture(0)
    while(1):
        # Capture frame from camera
        ret, frame = camera.read()
        
        frame=cv2.flip(frame,1)
        
        frame_original = np.copy(frame)
        
        cv2.line(frame ,(320, 0),(320, 480),(255,0,0),5)
        
        frame_1 = frame_original[:,:240]
        frame_2 = frame_original[:,240:]
                        
        
        
        if(time.time()-tzz>5):
            www = 1
        
        if www==0:
            cv2.putText(frame,"Player 1 show thumbs up in your panel to start game",(int(0.08*frame.shape[1]),int(0.97*frame.shape[0])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1,8) 
        
        if(www==1 and (v_initial==0) and (start==0 or start==2)):
            start = forExp2.recognise(frame_1, 2)
            if(start!=1):
                cv2.putText(frame,"Player 1 show thumbs up in your panel to start game",(int(0.08*frame.shape[1]),int(0.97*frame.shape[0])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1,8) 
            else:
                v_initial==1
                tz = time.time()
                
        if(start==1):
            #rounds = 5
            kz = str(5-(time.time()-tz))
            if rounds==0:
                cv2.putText(frame,"Player 1 show number of rounds you want to play within time left:"+kz,(int(0.05*frame.shape[1]),int(0.67*frame.shape[0])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1,8)
                if(time.time()-tz >= 5):
                    rounds = forExp2.recognise(frame_1,1)
                #rounds = 5
                    
            else:
              
                cv2.putText(frame,"Total Rounds"+str(rounds),(int(0.5*frame.shape[1]),int(0.07*frame.shape[0])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,20)
                c = c+1
                if(c==1):
                    t1 = time.time()
                    
                if(rounds1 <= rounds):
                    k = str(6-(time.time()-t1))
                    
                    cv2.putText(frame,"Seconds left until next round:"+k,(int(0.08*frame.shape[1]),int(0.97*frame.shape[0])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,8)
                    if(time.time()-t1 >= 6):
                        
                        rounds1 = rounds1 + 1
                        v1 = forExp2.recognise(frame_1, 3)
                        v2 = forExp2.recognise(frame_2, 3)    
                        
                        winner = rpsls.rpsls(v1, v2)
                        if winner !=0: 
                            if winner==1:
                                w1 = w1+1
                                if(w1>=(rounds/2)+1):
                                    rounds1 = rounds+1
                            else:
                                w2 = w2+1   
                                if(w2>=(rounds/2)+1):
                                    rounds1 = rounds+1
                        
                            
                        t1 = time.time()
                else:
                    if(w1>w2):
                        cv2.putText(frame,"Winner of the game is Player "+str(1)+"!!",(int(0.18*frame.shape[1]),int(0.27*frame.shape[0])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,20)
                    elif(w1<w2):
                        cv2.putText(frame,"Winner of the game is Player "+str(2)+"!!",(int(0.18*frame.shape[1]),int(0.27*frame.shape[0])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,20)
                    else:
                        cv2.putText(frame,"The game was a tie!!",(int(0.18*frame.shape[1]),int(0.27*frame.shape[0])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,20)
                
                
                    cv2.putText(frame,"Player 1 show L sign to close game or show thumbs up to replay",(int(0.08*frame.shape[1]),int(0.67*frame.shape[0])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1,20) 
                    
                    if(time.time()-t1>=10):
                        if(forExp2.recognise(frame_1, 2) == 2):
                            camera.release()
                            cv2.destroyAllWindows()
                            break
                        elif(forExp2.recognise(frame_1, 2) == 1):
                            camera.release()
                            cv2.destroyAllWindows()
                            two_players()
        if rounds1>1:
            cv2.putText(frame,"Player 1 choses:"+number_to_name(v1),(int(0.18*frame.shape[1]),int(0.77*frame.shape[0])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,20)
            cv2.putText(frame,"Player 2 choses:"+number_to_name(v2),(int(0.58*frame.shape[1]),int(0.77*frame.shape[0])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,20)
        
            if winner!=0:
                cv2.putText(frame,"Winner of last round is Player "+str(winner),(int(0.08*frame.shape[1]),int(0.55*frame.shape[0])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,20)
            else:
                cv2.putText(frame,"Last round was a tie",(int(0.08*frame.shape[1]),int(0.55*frame.shape[0])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,20)
            
        cv2.imshow("RPSLS Game Double Player", frame)
                    
        interrupt=cv2.waitKey(10)
        # Quit by pressing 'q'
        if  interrupt & 0xFF == ord('q'):
            camera.release()
            cv2.destroyAllWindows()
            break
'''    
  
camera = cv2.VideoCapture(0)
while(1):
    # Capture frame from camera
    ret, frame = camera.read()
    
    frame=cv2.flip(frame,1)
    rounds = forExp2.recognise(frame,1)
    print(rounds)
               
    interrupt=cv2.waitKey(10)
        # Quit by pressing 'q'
    if  interrupt & 0xFF == ord('q'):
        camera.release()
        cv2.destroyAllWindows()
        break
    
    cv2.imshow("rounds", frame)
camera.release()
cv2.destroyAllWindows()
     
'''