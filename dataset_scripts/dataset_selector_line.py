import sys
from os import listdir
from os.path import isfile, isdir
import numpy as np
import cv2

video = sys.argv[1]
cap = cv2.VideoCapture(video)
kernel = np.ones((10,10),np.uint8)
kernel_black = np.ones((10,10),np.uint8)
sensitivity = 0
lower_white = np.array([0,0,0])
upper_white = np.array([180,255,120])
while(True):
    # Capture frame-by-frame
    ret, img = cap.read()
    height, width, channels = img.shape
    img2 = img#[int(height*0):int(height*0.7), int(width*0.2):int(width*0.8)]
    hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    res = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_black)
    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('mask',mask)
    #cv2.imshow('res',res)
    cv2.imshow('image',img)
    k = cv2.waitKey(0)
    print(k)
    if k==82:
        print("forward")
        #cv2.imwrite('./sem/'+carpeta[2:-1]+ar,img)
    if k==81:
        print("left")
        #cv2.imwrite('./left_arrow/'+carpeta[2:-1]+ar,img)
    if k==83:
        print("right")
        #cv2.imwrite('./right_arrow/'+carpeta[2:-1]+ar,img)
    if k==32:
        continue;
    if k==113:
        break;
    cv2.destroyAllWindows()
# When everything done, release the capture
cap.release()
