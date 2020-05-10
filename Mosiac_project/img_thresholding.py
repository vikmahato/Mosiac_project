# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 19:47:04 2017

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

'''
capture_box_count=9
capture_box_dim=20
capture_box_sep_x=8
capture_box_sep_y=18
capture_pos_x=480
capture_pos_y=150
cap_region_x_begin=0.5 # start point/total width
cap_region_y_end=0.8 # start point/total width



def getting_hsv_histogram(frame_in,box_x,box_y):
    hsv = cv2.cvtColor(frame_in, cv2.COLOR_BGR2HSV)
    ROI = np.zeros([capture_box_dim*capture_box_count,capture_box_dim,3], dtype=hsv.dtype)
    for i in range(capture_box_count):
        ROI[i*capture_box_dim:i*capture_box_dim+capture_box_dim,0:capture_box_dim] = hsv[box_y[i]:box_y[i]+capture_box_dim,box_x[i]:box_x[i]+capture_box_dim]
    hand_hist = cv2.calcHist([ROI],[0, 1], None, [200, 256], [0, 200, 0, 256])
    cv2.normalize(hand_hist,hand_hist, 0, 255, cv2.NORM_MINMAX)
    return hand_hist
'''
    
# 2. Filters and threshold
def thresholding(frame_in,hand_hist):
    
    frame_in = cv2.medianBlur( frame_in, 3)
    hsv = cv2.cvtColor( frame_in, cv2.COLOR_BGR2HSV)
    back_projection = cv2.calcBackProject( [hsv], [0,1], hand_hist, [0,200,0,256], 1)
    disc = cv2.getStructuringElement( cv2.MORPH_ELLIPSE, ( morph_elem_size, morph_elem_size))
    cv2.filter2D( back_projection, -1, disc, back_projection)
    back_projection = cv2.GaussianBlur( back_projection,( gaussian_ksize, gaussian_ksize), gaussian_sigma)
    back_projection = cv2.medianBlur( back_projection, median_ksize)
    ret, thresh = cv2.threshold( back_projection, hsv_thresh_lower, 255, 0)
    
    #thresh1 = thresh
    #thresh = cv2.merge((thresh,thresh, thresh))
    
    #res = cv2.bitwise_and(frame_in, thresh)
    
    return thresh
  
    
def roi(frame_original):
    
    
    #box_pos_x=np.array([capture_pos_x,capture_pos_x+capture_box_dim+capture_box_sep_x,capture_pos_x+2*capture_box_dim+2*capture_box_sep_x,capture_pos_x,capture_pos_x+capture_box_dim+capture_box_sep_x,capture_pos_x+2*capture_box_dim+2*capture_box_sep_x,capture_pos_x,capture_pos_x+capture_box_dim+capture_box_sep_x,capture_pos_x+2*capture_box_dim+2*capture_box_sep_x],dtype=int)
    #box_pos_y=np.array([capture_pos_y,capture_pos_y,capture_pos_y,capture_pos_y+capture_box_dim+capture_box_sep_y,capture_pos_y+capture_box_dim+capture_box_sep_y,capture_pos_y+capture_box_dim+capture_box_sep_y,capture_pos_y+2*capture_box_dim+2*capture_box_sep_y,capture_pos_y+2*capture_box_dim+2*capture_box_sep_y,capture_pos_y+2*capture_box_dim+2*capture_box_sep_y],dtype=int)
    
    hand_histogram=np.loadtxt("hand_histogram.csv", delimiter=",", dtype = 'float32')
    #hand_histogram=getting_hsv_histogram(frame_original,box_pos_x,box_pos_y)
    
    areaArray = []
    thresh = thresholding( frame_original, hand_histogram)
    #cv2.imshow('Hand Gesture Recognition v5.0',thresh2)
    contour_frame = np.copy(thresh)
    _, contours, hierarchy = cv2.findContours( contour_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        areaArray.append(area)
    
    
    sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)
    
    #find the nth largest contour [n-1][1], in this case 2
    largestcontour = sorteddata[0][1]
            
    #cv2.drawContours(frame_original, [largestcontour], 0, (0,255,0), 3)
    
    (x, y, w, h) = cv2.boundingRect(largestcontour)
    roi = frame_original[y:y+h, x:x+w]
    #gray= cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    roi = cv2.resize( roi, (152, 152), interpolation = cv2.INTER_CUBIC)
    #roi, thresh2 = thresholding(roi, hand_histogram)
    gray = cv2.cvtColor( roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur( gray, (5,5), 0)
    ret, roi = cv2.threshold( blur, 10, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    
    # Display frame in a window
    #cv2.imshow('Hand Gesture',frame_original)
    #cv2.imshow('Hand Gesture2',roi)
    
    return roi
   