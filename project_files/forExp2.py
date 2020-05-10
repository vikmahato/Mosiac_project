# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 03:21:06 2017

@author: pc
"""


import numpy as np
import img_thresholding
import machine_learning_part as ML


def recognise(frame, model_selection):
        
    roi = img_thresholding.roi(frame)
    
    roi = np.reshape(roi, [1, -1]) 
    
    class_predicted = ML.prediction(roi, model_selection)
    
    return class_predicted
        
       
    
    