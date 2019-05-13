import numpy as np
import cv2

#cap = cv2.VideoCapture('http://192.168.0.104:8080/?action=stream')
imagen = cv2.imread('opencv_logo.png')
hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('lineVideo.mp4')
#Rango de colores detectados:
#Verdes:
verde_bajos = np.array([49,50,50])
verde_altos = np.array([107, 255, 255])
#Azules:
azul_bajos = np.array([100,65,75], dtype=np.uint8)
azul_altos = np.array([130, 255, 255], dtype=np.uint8)
#Rojos:
rojo_bajos1 = np.array([0,65,75], dtype=np.uint8)
rojo_altos1 = np.array([12, 255, 255], dtype=np.uint8)
rojo_bajos2 = np.array([240,65,75], dtype=np.uint8)
rojo_altos2 = np.array([256, 255, 255], dtype=np.uint8)

#Crear las mascaras
mascara_verde = cv2.inRange(hsv, verde_bajos, verde_altos)
mascara_rojo1 = cv2.inRange(hsv, rojo_bajos1, rojo_altos1)
mascara_rojo2 = cv2.inRange(hsv, rojo_bajos2, rojo_altos2)
mascara_azul = cv2.inRange(hsv, azul_bajos, azul_altos)

#Juntar todas las mascaras
mask = cv2.add(mascara_rojo1, mascara_rojo2)
mask = cv2.add(mask, mascara_verde)
mask = cv2.add(mask, mascara_azul)


fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('lineVideoFiltered.mp4', fourcc, 20.0, (640,480))


ret, imagen = cap.read()

kernel = np.ones((5,5),np.uint8)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #frame = cv2.addWeighted(frame,0.5,imagen,0.5,0)

    ###################################################
    # LINE FOLLOWER

    #img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    '''img = cv2.blur(img, (5, 5))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    thresh0 = cv2.adaptiveThreshold(s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    thresh1 = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    thresh2 = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    thresh = cv2.bitwise_or(thresh0, thresh1)
    '''
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of white color in HSV
    # change it according to your need !
    #lower_white = np.array([0,0,0], dtype=np.uint8)
    #upper_white = np.array([0,0,255], dtype=np.uint8)

    sensitivity = 15
    lower_white = np.array([0,0,255-sensitivity])
    upper_white = np.array([255,sensitivity,255])

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)
    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)
    res = cv2.erode(res,kernel,iterations = 1)
    #res = cv2.dilate(res,kernel,iterations = 1)
    #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) # Drawing color points requires RGB image
    # ret, thresh = cv2.threshold(frame, 105, 255, cv2.THRESH_BINARY)
    #tresh = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

    #signed_thresh = thresh[start_height].astype(np.int16) # select only one row
    #diff = np.diff(signed_thresh)   #The derivative of the start_height line

    ###################################################


    # Display the resulting frame
    #cv2.imshow('frame',frame)
    out.write(res)
    cv2.imshow('frame',res)
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
