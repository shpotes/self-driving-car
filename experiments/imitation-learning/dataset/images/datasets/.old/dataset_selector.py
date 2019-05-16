import sys
from os import listdir
from os.path import isfile, isdir
import numpy as np
import cv2

carpeta = sys.argv[1]

def ls(path):
    return [obj for obj in listdir(path) if isfile(path + obj)]

archivos = ls(carpeta)

print(archivos)
for ar in archivos:
    img = cv2.imread(carpeta+ar,1)
    cv2.imshow('image',img)
    k = cv2.waitKey(0)
    print(k)
    if k==82:
        print("sem")
        cv2.imwrite('./sem/'+carpeta[2:-1]+ar,img)
    if k==84:
        print("stop")
        cv2.imwrite('./stop/'+carpeta[2:-1]+ar,img)
    if k==81:
        print("left_arrow")
        cv2.imwrite('./left_arrow/'+carpeta[2:-1]+ar,img)
    if k==83:
        print("right_arrow")
        cv2.imwrite('./right_arrow/'+carpeta[2:-1]+ar,img)
    if k==32:
        print("person")
        cv2.imwrite('./person/'+carpeta[2:-1]+ar,img)
    if k==10:
        continue;
    if k==113:
        break;
    cv2.destroyAllWindows()
