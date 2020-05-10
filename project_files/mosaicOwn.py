# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 08:23:11 2017

@author: pc
"""

import cv2
import numpy as np
import math, time


hsv_thresh_lower=180
gaussian_ksize=11
gaussian_sigma=0
morph_elem_size=12
median_ksize=3
capture_box_count=9
capture_box_dim=20
capture_box_sep_x=8
capture_box_sep_y=18
capture_pos_x=480
capture_pos_y=150
cap_region_x_begin=0.5 # start point/total width
cap_region_y_end=0.8 # start point/total width
#first_iteration=True


def hand_capture(frame_in,box_x,box_y):
    hsv = cv2.cvtColor(frame_in, cv2.COLOR_BGR2HSV)
    ROI = np.zeros([capture_box_dim*capture_box_count,capture_box_dim,3], dtype=hsv.dtype)
    for i in range(capture_box_count):
        ROI[i*capture_box_dim:i*capture_box_dim+capture_box_dim,0:capture_box_dim] = hsv[box_y[i]:box_y[i]+capture_box_dim,box_x[i]:box_x[i]+capture_box_dim]
    hand_hist = cv2.calcHist([ROI],[0, 1], None, [200, 256], [0, 200, 0, 256])
    cv2.normalize(hand_hist,hand_hist, 0, 255, cv2.NORM_MINMAX)
    return hand_hist

    

    
# 2. Filters and threshold
def hand_threshold(frame_in,hand_hist):
    frame_in=cv2.medianBlur(frame_in,3)
    hsv=cv2.cvtColor(frame_in,cv2.COLOR_BGR2HSV)
    #hsv[0:int(cap_region_y_end*hsv.shape[0]),0:int(cap_region_x_begin*hsv.shape[1])]=0 # Right half screen only
    #hsv[int(cap_region_y_end*hsv.shape[0]):hsv.shape[0],0:hsv.shape[1]]=0
    back_projection = cv2.calcBackProject([hsv], [0,1],hand_hist, [0,200,0,256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_elem_size,morph_elem_size))
    cv2.filter2D(back_projection, -1, disc, back_projection)
    back_projection=cv2.GaussianBlur(back_projection,(gaussian_ksize,gaussian_ksize), gaussian_sigma)
    #back_projection=cv2.medianBlur(back_projection,median_ksize)
    ret, thresh = cv2.threshold(back_projection, hsv_thresh_lower, 255, 0)
    
    
    thresh1 = thresh
    thresh = cv2.merge((thresh,thresh, thresh))
    #cv2.imshow('Hand Gesture Recognition v2.0',thresh)
    
    res = cv2.bitwise_and(frame_in, thresh)
    #thresh1 = find_contour(thresh1)
    
    return res, thresh1
  
    

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
    gray=np.copy(frame)
    
    if (not (capture_done)):
        cv2.putText(frame_original,"Place hand inside boxes and press 'c' to capture hand histogram",(int(0.08*frame.shape[1]),int(0.97*frame.shape[0])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1,8)
        first_iteration=True
        #box_pos_x=np.array([capture_pos_x,capture_pos_x+capture_box_dim+capture_box_sep_x,capture_pos_x+2*capture_box_dim+2*capture_box_sep_x,capture_pos_x+3*capture_box_dim+3*capture_box_sep_x,capture_pos_x,capture_pos_x+capture_box_dim+capture_box_sep_x,capture_pos_x+2*capture_box_dim+2*capture_box_sep_x,capture_pos_x+3*capture_box_dim+3*capture_box_sep_x,capture_pos_x,capture_pos_x+capture_box_dim+capture_box_sep_x,capture_pos_x+2*capture_box_dim+2*capture_box_sep_x,capture_pos_x+3*capture_box_dim+3*capture_box_sep_x,capture_pos_x,capture_pos_x+capture_box_dim+capture_box_sep_x,capture_pos_x+2*capture_box_dim+2*capture_box_sep_x,capture_pos_x+3*capture_box_dim+3*capture_box_sep_x,capture_pos_x,capture_pos_x+capture_box_dim+capture_box_sep_x,capture_pos_x+2*capture_box_dim+2*capture_box_sep_x,capture_pos_x+3*capture_box_dim+3*capture_box_sep_x],dtype=int)
        #box_pos_y=np.array([capture_pos_y,capture_pos_y,capture_pos_y,capture_pos_y,capture_pos_y+capture_box_dim+capture_box_sep_y,capture_pos_y+capture_box_dim+capture_box_sep_y,capture_pos_y+capture_box_dim+capture_box_sep_y,capture_pos_y+capture_box_dim+capture_box_sep_y,capture_pos_y+2*capture_box_dim+2*capture_box_sep_y,capture_pos_y+2*capture_box_dim+2*capture_box_sep_y,capture_pos_y+2*capture_box_dim+2*capture_box_sep_y,capture_pos_y+2*capture_box_dim+2*capture_box_sep_y,capture_pos_y+3*capture_box_dim+3*capture_box_sep_y,capture_pos_y+3*capture_box_dim+3*capture_box_sep_y,capture_pos_y+3*capture_box_dim+3*capture_box_sep_y,capture_pos_y+3*capture_box_dim+3*capture_box_sep_y,capture_pos_y+4*capture_box_dim+4*capture_box_sep_y,capture_pos_y+4*capture_box_dim+4*capture_box_sep_y,capture_pos_y+4*capture_box_dim+4*capture_box_sep_y,capture_pos_y+4*capture_box_dim+4*capture_box_sep_y],dtype=int)
        #box_pos_x=np.array([capture_pos_x,capture_pos_x+capture_box_dim+capture_box_sep_x,capture_pos_x+2*capture_box_dim+2*capture_box_sep_x,capture_pos_x,capture_pos_x+capture_box_dim+capture_box_sep_x,capture_pos_x+2*capture_box_dim+2*capture_box_sep_x,capture_pos_x,capture_pos_x+capture_box_dim+capture_box_sep_x,capture_pos_x+2*capture_box_dim+2*capture_box_sep_x],dtype=int)
        #box_pos_y=np.array([capture_pos_y,capture_pos_y,capture_pos_y,capture_pos_y+capture_box_dim+capture_box_sep_y,capture_pos_y+capture_box_dim+capture_box_sep_y,capture_pos_y+capture_box_dim+capture_box_sep_y,capture_pos_y+2*capture_box_dim+2*capture_box_sep_y,capture_pos_y+2*capture_box_dim+2*capture_box_sep_y,capture_pos_y+2*capture_box_dim+2*capture_box_sep_y],dtype=int)
    
        #for i in range(capture_box_count):
            #cv2.rectangle(frame_original,(box_pos_x[i],box_pos_y[i]),(box_pos_x[i]+capture_box_dim,box_pos_y[i]+capture_box_dim),(255,0,0),1)
    else:
        areaArray = []
        frame, thresh2 =hand_threshold(frame_original, hand_histogram)
        cv2.imshow('Hand Gesture Recognition v5.0',thresh2)
        contour_frame=np.copy(thresh2)
        _,contours,hierarchy=cv2.findContours(contour_frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            areaArray.append(area)
        
        #first sort the array by area
        sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)
        
        #find the nth largest contour [n-1][1], in this case 2
        largestcontour = sorteddata[0][1]
    
    
    
        #draw it
        hull = cv2.convexHull(largestcontour,returnPoints = False)
        defects = cv2.convexityDefects(largestcontour,hull)
       
        #for i in range(defects.shape[0]):
            #s,e,f,d = defects[i,0]
            #start = tuple(largestcontour[s][0])
            #end = tuple(largestcontour[e][0])
            #far = tuple(largestcontour[f][0])
            #cv2.line(frame_original,start,end,[0,255,0],2)
            #cv2.circle(frame_original,far,5,[0,0,255],-1)
                
                
        #b = cv2.drawContours(frame_original, [largestcontour], 0, (0,255,0), 3)
        (x, y, w, h) = cv2.boundingRect(largestcontour)
        roi = frame_original[y:y+h, x:x+w]
        #gray= cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        roi = cv2.resize( roi, (152, 152), interpolation = cv2.INTER_CUBIC)
        gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        ret, roi = cv2.threshold(blur,10,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) #to get roi as noiseless binary
        #roi, thresh3 = hand_threshold(roi, hand_histogram)
        #gray= cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        #sift = cv2.xfeatures2d.SIFT_create()
        #kp, des = sift.detectAndCompute(gray,None)
        
        #roi=cv2.drawKeypoints(gray,kp)
        #print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
        #mask = np.zeros(roi.shape,np.uint8)
        #mask[y:y+h,x:x+w] = roi[y:y+h,x:x+w]
        #mask = np.zeros(b.shape[:2], np.uint8)
        #mask[y:y+h, x:x+w] = 255
        #roi = cv2.bitwise_and(roi, roi, mask = mask)
        #cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
        
        #roi = np.zeros(roi.shape[:2], np.uint8)
        #cv2.drawContours(roi, largestcontour, -1, 255, -1)
        #mask = np.zeros(roi.shape,dtype="uint8")
        #cv2.drawContours(mask, [largestcontour], -1, 255, -1)
        #cv2.drawContours(mask, [largestcontour], 0, (0,255,0), 3)
        #cv2.fillConvexPoly(roi, [largestcontour])
        
        # Display frame in a window
    cv2.imshow('Hand Gesture Recognition v1.0',frame_original)
    cv2.imshow('Hand Gesture Recognition v1qv.0',roi)
            
    
    
    interrupt=cv2.waitKey(10)
    # Quit by pressing 'q'
    if  interrupt & 0xFF == ord('q'):
        break
    # Capture hand by pressing 'c'
    elif interrupt & 0xFF == ord('c'):
        capture_done=1
        #hand_histogram=hand_capture(frame_original,box_pos_x,box_pos_y)
        hand_histogram=np.loadtxt("hand_histogram.csv", delimiter=",", dtype = 'float32')
    # Reset captured hand by pressing 'r'
    elif interrupt & 0xFF == ord('r'):
        capture_done=0
    
    
# Release camera & end program
camera.release()
cv2.destroyAllWindows()