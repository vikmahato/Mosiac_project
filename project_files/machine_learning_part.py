# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 03:21:45 2017

@author: pc
"""

import cv2
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier as ET
import os.path as path
from sklearn.externals import joblib


def prediction(roi, model_selection):
    
    if(model_selection == 0):
        clf = joblib.load('Mosaic/model_for_players.pkl') 
    elif(model_selection == 1):
        clf = joblib.load('Mosaic/model_for_rounds2.pkl') 
    elif(model_selection == 2):
        clf = joblib.load('Mosaic/model_for_start_end.pkl') 
    elif(model_selection == 3):
        clf = joblib.load('Mosaic/model_for_gestures2.pkl') 
        
    predicted = clf.predict(roi)
    
    return predicted
    
    
    
    
'''
    train = 0
    for i in range(5):
        for j in range(1,601):
            if(path.isfile("C:/Users/pc/Desktop/Desktop2/Mosaic/Dataset/"+str(i)+"/"+str(j)+".jpg")):
                img = cv2.imread("C:/Users/pc/Desktop/Desktop2/Mosaic/Dataset/"+str(i)+"/"+str(j)+".jpg")
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5,5), 0)
                ret, img = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                
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
    
    predicted = model.predict(roi)
    
    return predicted


kf = KFold(n_splits = 10, shuffle = True)

c_score = cross_val_score(model, train, labels, cv = kf)

print("ET", np.mean(c_score))

filename = 'finalised_model_ET.pkl'
#pickle.dump(model, open(filename, 'wb'))

model = pickle.load(open(filename, 'rb'))
'''










