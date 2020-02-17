# -*- coding: utf-8 -*-


import cv2
import numpy as np


kamera = cv2.VideoCapture(0)  # bilgisayar kamerası kullanımı için 0
notebook = cv2.imread("ben.png", 0)  # 0 alınan resmi direkt olarak grey e çevirir
kalem = cv2.imread("kalem.png", 0)  # 0 alınan resmi direkt olarak grey e çevirir

while True:
    ret,frame = kamera.read()  # kameradan frame almak
    gri_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #  frame i griye çevirdim
    
    
    
    weight,height = notebook.shape    # alınan resmin eni ve boyu 
    
    res = cv2.matchTemplate(gri_frame,notebook,  cv2.TM_CCOEFF_NORMED )
    threshold_value = 0.8
    loc = np.where( res > threshold_value)
    
    for n in zip(*loc[::-1]):
        cv2.rectangle(frame , n ,(n[0]+height, n[1]+weight), (0,0,255),2)
        cv2.putText( frame , "Hayvan", (n[0]+height, n[1]+weight),cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,0),1)
    
    weight,height = kalem.shape    # alınan resmin eni ve boyu 
    
    res = cv2.matchTemplate(gri_frame,kalem,  cv2.TM_CCOEFF_NORMED )
    threshold_value = 0.8
    loc = np.where( res > threshold_value)
    
    for n in zip(*loc[::-1]):
        cv2.rectangle(frame , n ,(n[0]+height, n[1]+weight), (0,0,255),2)
        cv2.putText( frame , "Kalem", (n[0]+height, n[1]+weight),cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,0),1)
    
    
    cv2.imshow("Frame", frame)
    
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


kamera.release()
cv2.destroyAllWindows()
    
