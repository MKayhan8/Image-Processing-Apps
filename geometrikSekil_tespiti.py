# -*- coding: utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("sekiller.jpg")

gri_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #  frame i griye çevirdim

cv2.imshow("Shapes", img)


#ret,thresh0 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

ret,thresh1 = cv2.threshold(gri_image,240,255,cv2.THRESH_BINARY)
cv2.imshow("Threshold", thresh1)

#median = cv2.medianBlur(thresh1,5)
#cv2.imshow("Median", median)
#
#kernel = np.ones((5,5),np.uint8)
#yayma = cv2.dilate(median,kernel,iterations = 1)
#cv2.imshow("Morfo- yayılım", yayma)

contours,hierachy=cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
font = cv2.FONT_HERSHEY_COMPLEX
for cnt in contours:
    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.01 * perimeter, True)  
    cv2.drawContours( img , [approx] , 0,(255,0,0,), 3 )
    print("approx vertices:", len(approx))
    x = approx.ravel()[0]           # gelen approx koordinatlarının ilk x ve y sine şeklin adını yazdırıyoruz
    y = approx.ravel()[1]
    if len(approx) == 3:
        cv2.putText(img , "Triangle", (x , y), font , 1,(0))
    elif len(approx) == 4:
        cv2.putText(img , "Rectangle", (x , y), font , 1,(0))
    elif len(approx) == 5:
        cv2.putText(img , "Pentagon", (x , y), font , 1,(0))
    elif len(approx) == 6:
        cv2.putText(img , "Hexagon", (x , y), font , 1,(0))
    elif len(approx) == 7:
        cv2.putText(img , "Heptagon", (x , y), font , 1,(0))
    elif len(approx) == 8:
        cv2.putText(img , "Octagon", (x , y), font , 1,(0))
    elif 8 < len(approx) <15 :
        cv2.putText(img , "Elips", (x , y), font , 1,(0))
    else:
        cv2.putText(img , "Circle", (x , y), font , 1,(0))
    
    

        
   
#    cv2.drawContours( img , [cnt] , -1,(0,255,0), 5 )
cv2.imshow("conturlu", img)


#ret,thresh2 = cv2.threshold(img,242,255,cv2.THRESH_BINARY_INV)
#ret,thresh3 = cv2.threshold(img,242,255,cv2.THRESH_TRUNC)
#ret,thresh4 = cv2.threshold(img,242,255,cv2.THRESH_TOZERO)
#ret,thresh5 = cv2.threshold(img,242,255,cv2.THRESH_TOZERO_INV)

