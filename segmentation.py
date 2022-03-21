import mahotas
import mahotas.demos
import mahotas.features
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils
import os
from numpy.lib.type_check import imag
from scipy.spatial import distance as dist
import glob 
import pandas as pd




def zernike_shape_descriptors(image):
    img0 =cv2.imread(image)
    #img0 =cv2.resize(img0 ,(600,500))
    img = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

    #You can try and choose among those :
    
    #ret0 ,th0 = cv2.threshold(img , 254 ,255 , cv2.THRESH_BINARY_INV)
    ret1, th1 = cv2.threshold(img , 30 ,255 , cv2.THRESH_BINARY)
    #th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    #th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
    #th4 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    #th5 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)

    #th1 = cv2.dilate(th1, None, iterations=7)
    #th1 = cv2.erode(th1, None, iterations=2)

    countours ,hierarchy = cv2.findContours(th1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    #cv2.drawContours(img0, countours, -1 , (0 ,255,0,0 ),3)

    max_area = 0
    for i in range(len(countours)):
        cnt = countours[i]
        area = cv2.contourArea(cnt)
        #print(area)
        if (area > max_area):
            max_area = area
            ci = i

    largest_areas = sorted(countours , key= cv2.cv2.contourArea)
    mask= np.zeros(np.shape(img0),np.uint8)
    img_countours = cv2.drawContours(mask , [largest_areas[-1]] , 0, (255,255,255,255),-1)

   
    (x,y),radius = cv2.minEnclosingCircle(largest_areas[-1])
    center = (int(x),int(y))
    radius = int(radius)



    #if ((radius>np.shape(img)[0]/2) or (radius>np.shape(img)[1])/2 ):
    #radius = min (np.shape(img)[0]//2,np.shape(img)[1]//2)
    

    #print(radius) 
    #print(np.shape(img)[0])
    #print(np.shape(img)[1])

    Descriptors = []


    cv2.circle(img_countours,center,radius,(0,0,255),2)

    Descriptors=(list(mahotas.features.zernike_moments(img_countours[:,:,0], radius , degree=7)))

    #cv2.imshow('img',img0)
    #cv2.imshow('mask',mask)
    #cv2.imshow('countours',img_countours)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return Descriptors








#print(tuple(shape_descriptors('C:/Users/HP/OneDrive/Bureau/pfe/coil-100/boites/obj1__5.png')))
#RETOURNER UNE LISTE DE 8 LISTES DE DESCRIPTEURS

