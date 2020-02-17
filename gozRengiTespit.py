# -*- coding: utf-8 -*-


import cv2  #opencv importing
import numpy as np  # matrisler üzerinde işlem

# yüz tespit eden cascade tanımlama
face_casc = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# göz tespit eden cascade tanımlama
eye_casc = cv2.CascadeClassifier("haarcascade_eye.xml")


# istenilen resmi tanımlama

#orgResim = cv2.imread("kahverengiGozluYuz.jpg")
orgResim = cv2.imread("yesil.jpg")
#orgResim = cv2.imread("siyahiMavi.jpg")
#orgResim = cv2.imread("beyazTenliKahverengiGoz.jpg")
#orgResim = cv2.imread("mavi.jpg")
#orgResim = cv2.imread("siyahGoz.jpg")



#cv2.imshow("Orijinal foto", orgResim)  # show image
scale_percent = 50 # percent of original size
width = int(orgResim.shape[1] * scale_percent / 100)
height = int(orgResim.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(orgResim, dim, interpolation = cv2.INTER_AREA)
orgResim = resized.copy()
cv2.imshow("Resized image", orgResim) 

#  alınan resmi griye çevirme
greyResim = cv2.cvtColor(orgResim, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Orijinal foto", greyResim)  # show image

faces = face_casc.detectMultiScale(greyResim, 1.1 , 4 )



for ( x, y, w, h) in faces:
    cv2.rectangle( orgResim , (x,y), (x+w, y+h), (0,0,255), 2 )
    faceRegion = orgResim[y:y+h , x:x+w ]
    greyFaceRegion = cv2.cvtColor(faceRegion, cv2.COLOR_BGR2GRAY)
    eyes = eye_casc.detectMultiScale(greyFaceRegion, 1.1 , 4 )
    faceRegionCopy= faceRegion.copy()
    
    for ( a, b, c, d) in eyes:
        cv2.rectangle( faceRegion , (a,b), (a+c, b+d), (255,0,0), 2 )
        eyeRegion = faceRegionCopy[b:b+d , a:a+c ]
        #cv2.imshow("Eyes",eyeRegion)
        

        


        
    
    
#cv2.imshow("Detected Faces",orgResim)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#---------------- Blue HSV Range -------------------
def blue():
    lower_blue = np.array([80,50,50])
    upper_blue = np.array([130,255,255])
    hsv = cv2.cvtColor(eyeRegion, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange( hsv , lower_blue, upper_blue )
    finalResim = cv2.bitwise_and( hsv , hsv, mask= mask)
    cv2.imshow("mask ",mask)
    print(mask.size)
    print(mask.shape)
    cv2.imshow("final", finalResim)
    return  ( ( np.sum(mask)/255 ) /mask.size )  * 2000 #  oranlar güzelleşsin diye

#------------------ Green HSV Range ------------
def green():
    lower_green = np.array([37,10,10])
    upper_green = np.array([65,255,255])
    hsv = cv2.cvtColor(eyeRegion, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange( hsv , lower_green, upper_green )
    finalResim = cv2.bitwise_and( hsv , hsv, mask= mask)
    cv2.imshow("mask ",mask)
#    print(mask.shape)
    print(mask)
    #print(finalResim)
    cv2.imshow("final", finalResim)
    return  ( ( np.sum(mask)/255 ) /mask.size )  * 2000

#------------------Brown HSV Range -----------------
def brown():
    lower_brown = np.array([2, 100, 65])
    upper_brown = np.array ([12,170,255])
    hsv = cv2.cvtColor(eyeRegion, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange( hsv , lower_brown, upper_brown )
    finalResim = cv2.bitwise_and( hsv , hsv, mask= mask)
    cv2.imshow("mask ",mask)
    print(mask.shape)
    print(finalResim)
    cv2.imshow("final", finalResim)
    return  ( ( np.sum(mask)/255 ) /mask.size ) * 2000
#-------------------------------------------------



if blue() > 50 :
    print( "BLUE COLOR EYE" )
elif green() > 50 :
    print( "GREEN COLOR EYE" )
elif brown() > 50:
    print( "BROWN COLOR EYE" )
    
#cv2.imshow("eyeRegion",eyeRegion)
cv2.imshow("Oorg Resim",orgResim)
#cv2.imshow("HSV Eyes",hsv)
#cv2.imshow("mask ",mask)
#cv2.imshow("final", finalResim)
#cv2.imshow("bumbum",bumbum)
#cv2.imshow("faceRegion",faceRegion)

# conda install -c conda-forge opencv

    


