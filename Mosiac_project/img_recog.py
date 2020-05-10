# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 17:18:10 2017

@author: pc
"""

import cv2
import numpy as np
import math, time
import img_thresholding
#from sklearn.neural_network import MLPClassifier as MLP
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import ExtraTreesClassifier as ET
#from sklearn.svm import SVC
import os.path as path
import rpsls
import pickle
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score

t2 = time.time()


hsv_thresh_lower=180
gaussian_ksize=11
gaussian_sigma=0
morph_elem_size=12
median_ksize=3


def thresholding(frame_in,hand_hist):
    frame_in=cv2.medianBlur(frame_in,3)
    hsv=cv2.cvtColor(frame_in,cv2.COLOR_BGR2HSV)
    back_projection = cv2.calcBackProject([hsv], [0,1],hand_hist, [0,200,0,256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_elem_size,morph_elem_size))
    cv2.filter2D(back_projection, -1, disc, back_projection)
    back_projection=cv2.GaussianBlur(back_projection,(gaussian_ksize,gaussian_ksize), gaussian_sigma)
    back_projection=cv2.medianBlur(back_projection,median_ksize)
    ret, thresh = cv2.threshold(back_projection, hsv_thresh_lower, 255, 0)
    
    thresh1 = thresh
    thresh = cv2.merge((thresh,thresh, thresh))
    
    res = cv2.bitwise_and(frame_in, thresh)
    
    return res, thresh1
   
'''
train = 0
for i in range(5):
    for j in range(1,260):
        if(path.isfile("C:/Users/pc/Desktop/Desktop2/Mosaic/Dataset/"+str(i)+"/"+str(j)+".jpg")):
            img = cv2.imread("C:/Users/pc/Desktop/Desktop2/Mosaic/Dataset/"+str(i)+"/"+str(j)+".jpg")
            #hand_histogram=np.loadtxt("hand_histogram.csv", delimiter=",", dtype = 'float32')
            #img, thresh2 = thresholding(img, hand_histogram)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray,(5,5),0)
            ret, img = cv2.threshold(blur,10,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            
            img = np.reshape(img, [-1])
            #print(img.shape)
            if(i==0 and j==1):
                labels = i
                train = img
            else:      
                labels = np.append(labels, i)
                train = np.append(train, img)

train = np.reshape(train, [-1, 152*152])

print(train.shape)
print(labels.shape)

model = ET()
model.fit(train, labels)


kf = KFold(n_splits = 10, shuffle = True)

c_score = cross_val_score(model, train, labels, cv = kf)

print("LR", np.mean(c_score))

filename = 'finalised_model_ET.pkl'
#pickle.dump(model, open(filename, 'wb'))


model = pickle.load(open(filename, 'rb'))

'''
print(time.time()-t2)

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
         cv2.putText(frame_original,"Press 'c' when you are ready",(int(0.08*frame.shape[1]),int(0.97*frame.shape[0])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1,8)
         
         #box_pos_x=np.array([capture_pos_x,capture_pos_x+capture_box_dim+capture_box_sep_x,capture_pos_x+2*capture_box_dim+2*capture_box_sep_x,capture_pos_x+3*capture_box_dim+3*capture_box_sep_x,capture_pos_x,capture_pos_x+capture_box_dim+capture_box_sep_x,capture_pos_x+2*capture_box_dim+2*capture_box_sep_x,capture_pos_x+3*capture_box_dim+3*capture_box_sep_x,capture_pos_x,capture_pos_x+capture_box_dim+capture_box_sep_x,capture_pos_x+2*capture_box_dim+2*capture_box_sep_x,capture_pos_x+3*capture_box_dim+3*capture_box_sep_x,capture_pos_x,capture_pos_x+capture_box_dim+capture_box_sep_x,capture_pos_x+2*capture_box_dim+2*capture_box_sep_x,capture_pos_x+3*capture_box_dim+3*capture_box_sep_x,capture_pos_x,capture_pos_x+capture_box_dim+capture_box_sep_x,capture_pos_x+2*capture_box_dim+2*capture_box_sep_x,capture_pos_x+3*capture_box_dim+3*capture_box_sep_x],dtype=int)
         #box_pos_y=np.array([capture_pos_y,capture_pos_y,capture_pos_y,capture_pos_y,capture_pos_y+capture_box_dim+capture_box_sep_y,capture_pos_y+capture_box_dim+capture_box_sep_y,capture_pos_y+capture_box_dim+capture_box_sep_y,capture_pos_y+capture_box_dim+capture_box_sep_y,capture_pos_y+2*capture_box_dim+2*capture_box_sep_y,capture_pos_y+2*capture_box_dim+2*capture_box_sep_y,capture_pos_y+2*capture_box_dim+2*capture_box_sep_y,capture_pos_y+2*capture_box_dim+2*capture_box_sep_y,capture_pos_y+3*capture_box_dim+3*capture_box_sep_y,capture_pos_y+3*capture_box_dim+3*capture_box_sep_y,capture_pos_y+3*capture_box_dim+3*capture_box_sep_y,capture_pos_y+3*capture_box_dim+3*capture_box_sep_y,capture_pos_y+4*capture_box_dim+4*capture_box_sep_y,capture_pos_y+4*capture_box_dim+4*capture_box_sep_y,capture_pos_y+4*capture_box_dim+4*capture_box_sep_y,capture_pos_y+4*capture_box_dim+4*capture_box_sep_y],dtype=int)
         #box_pos_x=np.array([capture_pos_x,capture_pos_x+capture_box_dim+capture_box_sep_x,capture_pos_x+2*capture_box_dim+2*capture_box_sep_x,capture_pos_x,capture_pos_x+capture_box_dim+capture_box_sep_x,capture_pos_x+2*capture_box_dim+2*capture_box_sep_x,capture_pos_x,capture_pos_x+capture_box_dim+capture_box_sep_x,capture_pos_x+2*capture_box_dim+2*capture_box_sep_x],dtype=int)
         #box_pos_y=np.array([capture_pos_y,capture_pos_y,capture_pos_y,capture_pos_y+capture_box_dim+capture_box_sep_y,capture_pos_y+capture_box_dim+capture_box_sep_y,capture_pos_y+capture_box_dim+capture_box_sep_y,capture_pos_y+2*capture_box_dim+2*capture_box_sep_y,capture_pos_y+2*capture_box_dim+2*capture_box_sep_y,capture_pos_y+2*capture_box_dim+2*capture_box_sep_y],dtype=int)
    
         #for i in range(capture_box_count):
             #cv2.rectangle(frame_original,(box_pos_x[i],box_pos_y[i]),(box_pos_x[i]+capture_box_dim,box_pos_y[i]+capture_box_dim),(255,0,0),1)
     else:
         k = str(1-(time.time()-t1))
         cv2.putText(frame_original,k,(int(0.08*frame.shape[1]),int(0.97*frame.shape[0])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1,8)
         if(time.time()-t1>=1):
             roi = img_thresholding.roi(frame_original)
             camera.release()
             cv2.destroyAllWindows()
             break
     
     cv2.imshow('Hand Gesture',frame_original)
     
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

         
#roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#cv2.imwrite("abd.jpg", roi)
#cv2.imwrite("roi.jpg", roi)
roi = np.reshape(roi, [1, -1]) 

predicted = model.predict(roi)

if predicted==0:
    print("Rock")
elif predicted==1:
    print("Spock")
elif predicted==2:
    print("Paper")
elif predicted==3:
    print("Lizard")
elif predicted==4:
    print("Scissor")
    


    

































