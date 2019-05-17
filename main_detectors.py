from keras.preprocessing.image import img_to_array
from keras.models import load_model
import argparse
#import io
from skimage.feature import hog
from functools import reduce
import requests
import time
import cv2
import numpy as np
import time
import dlib
import sys
import os
from skimage import io
from skimage import data, color, exposure

def send_action(action):
    run_action(action)

def run_action(base_url, action):
    url = base_url + 'run/?action=' + str(action)
    print(url)
    __request__(url)

def __request__(url, times=10):
    for _ in range(times):
        try:
            requests.get(url)
            return 0
        except Exception as e:
            print('connection error, try again')
            # print(str(e))
    print('abort')
    return -1
    # pass

def connection_ok(base_url):
    """Check whetcher connection is ok

    Post a request to server, if connection ok, server will return http response 'ok'

    Args:
        none

    Returns:
        if connection ok, return True
        if connection not ok, return False

    Raises:
        none
    """
    cmd = 'connection_test' + "/"
    url = base_url + cmd
    print('url: %s'% url)
    # if server find there is 'connection_test' in request url, server will response 'Ok'
    try:
        r=requests.get(url)
        if r.text == 'OK':
            return True
    except:
        return False

def detect_arrow(img):

    #_, threshold = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
    #contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_gray = np.array([0, 0, 0], np.uint8)
    upper_gray = np.array([180, 255, 80], np.uint8)
    mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)
    mask = cv2.inRange(hsv, lower_gray, upper_gray )
    np.save('')
    fd, hog_image = hog(mask, orientations=8, pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1), visualize=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    cv2.imshow("mask", mask)
    cv2.imshow("mask_hog", hog_image_rescaled)
    img_res = cv2.bitwise_and(img, img, mask = mask)
    print(mask.shape)
    h,w = mask.shape
    left_image = mask[0:(h), 0:(w//2)].copy()
    cv2.imshow("left_image", left_image)
    right_image = mask[0:h, w//2:(w)].copy()
    cv2.imshow("right_image", right_image)

    triangle_n = 0
    square_n = 0
    right_a = False

    # Find if the arrow is in the right image
    cnts,  _ = cv2.findContours(right_image.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
    	peri = cv2.arcLength(cnt, True)
    	approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    	#print("lenApprox")
    	#print(len(approx))
    	# Is arrow
    	if len(approx)==7:
    		#cv2.drawContours(img,[cnt],0,(0,255,0),-1)
    		right_a = True


    #Left
    cnts,  _ = cv2.findContours(left_image.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
    	peri = cv2.arcLength(cnt, True)
    	approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

    	#print("lenApproxL")
    	#print(len(approx))
    	if len(approx)==7:
    		#cv2.drawContours(img,[cnt],0,(0,255,0),-1)
    		right_a = False

    print('r' if right_a else 'l')
    return right_a


def detect_distance(self, output,w, h, img):

		decision = -2
		for i in output:
			box = i[2]
			y1 = int(box[0] * w)
			x1 = int(box[1] * h)
			y2 = int(box[2] * w)
			x2 = int(box[3] * h)
			#print("score"+str(i[0]))
			#print("pred"+str(i[1]))
			arrow = self.detect_arrow(img[y1:y1+(y2-y1), x1:x1+(x2-x1)])
			label = ""
			if (i[1]==1):
				label="person"
			elif ("right" in arrow):
				label="right_arrow"
			elif ("left" in arrow):
				label="left_arrow"
			#elif (i[1]==2):
			#	label="right_arrow"
			#elif (i[1]==3):
			#	label="left_arrow"
			elif (i[1]==4):
				label="sem"
			elif (i[1]==5):
				label="stop"
			#print(y1,x1,y2,x2)
			#dist = utils.distanceObject(x2-x1,label)
			#print("dist")
			#print(dist)

			# check if it's inside the red area
			xmiddle = x1 + ((x2 - x1)/2)
			ymiddle = y1 + ((y2 - y1)/2)
			inside_red_box = False
			if xmiddle >= 116 and xmiddle <= 549 and ymiddle >= 111 and ymiddle <= 262:
				inside_red_box =  True
				if "stop" in label or "sem" in label or "person" in label:
					#decision = "stop"
					decision = -1
					return decision
				if "left_arrow" in label:
					#decision = "forward_left"
					decision = 0
					return decision
				if "right_arrow" in label:
					#decision = "forward_right"
					decision = 2
					return decision


			#print("inside")
			#print(inside_red_box)
		return decision

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'ip', help='ip for raspberry pi car')
    args = parser.parse_args()
    ip = args.ip # 10.42.0.235
    port = '8000'
    base_url = 'http://' + ip + ':'+ port + '/'

    actions = ['stop', 'forward_left', 'forward', 'forward_right', 'backward_left', 'backward', 'backward_right', 'fwstraight']

    #if(not connection_ok(base_url)):
    #    raise Exception('Connection failed')

    detector1 = dlib.fhog_object_detector("./../detector_stop.svm")
    detector2 = dlib.fhog_object_detector("./../detector_sem.svm")
    detector3 = dlib.fhog_object_detector("./../detector_der.svm")
    #detector4 = dlib.fhog_object_detector("./../detector_izq.svm")
    detectors = [detector1, detector2]
    detectors2 = [detector3]
    #cap = cv2.VideoCapture(base_url + '?action=stream')
    #cap = cv2.VideoCapture('http://10.42.0.235:8080/?action=stream')
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('./../linesData2.mp4')
    ret, frame = cap.read()

    if(not ret):
        raise Exception('no image read')

    model = load_model("./lineModels/anonymous_modelv7.h5")
    nFrame = 0
    wait = 0.2
    waitFrames = 40 # Number of frames to ignore
    while(ret): # and (nFrame<20)):
        # Capture frame-by-frame
        #print(nFrame)
        boxes_a, confidences, detector_idxs = dlib.fhog_object_detector.run_multiple(detectors2, frame, upsample_num_times=1, adjust_threshold=0.0)
        if(len(boxes_a) != 0):
            d = boxes_a[0]
            arrow_bb = frame[d.top():d.bottom(), d.left():d.right()]
            dist = arrow_bb.shape[0]
            print(dist)
            #cv2.imshow("arrow", arrow_bb)
            detect_arrow(arrow_bb)
        #detect_arrow(frame)
        boxes, confidences, detector_idxs = dlib.fhog_object_detector.run_multiple(detectors, frame, upsample_num_times=1, adjust_threshold=0.0)
        # for i, d in enumerate(boxes):
        #     print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        #         i, d.left(), d.top(), d.right(), d.bottom()))
        #     cv2.rectangle(frame,(d.left(),d.top()),(d.right(),d.bottom()),(0,255,0),3)

        image = frame#.array
        image = cv2.resize(image, (56, 56), interpolation = cv2.INTER_AREA)
        cv2.imshow("frame", frame)
        image = img_to_array(image)
        image = np.array(image, dtype="float") / 255.0
        image = image.reshape(-1, 56, 56, 3)


        for i in range(waitFrames):
            _,_ = cap.read()

        i_time = time.time()

        '''if(len(boxes) == 0):
            prediction = np.argmax(model.predict(image))
            print('Time predicting action: {}'.format(time.time()-i_time))
            print("prediction = {} ... action = {}".format(prediction, actions[prediction+1]))
            if prediction == 0 :#and len(boxes) == 0:
                print("forward_left")
                run_action(base_url, actions[1])
                time.sleep(wait)
                run_action(base_url, actions[0])
                run_action(base_url, actions[-1])
            elif prediction == 1:# and len(boxes) == 0:
                print("forward")
                run_action(base_url, actions[2])
                time.sleep(wait)
                run_action(base_url, actions[0])
            elif prediction == 2:# and len(boxes) == 0:
                print("forward_right")
                run_action(base_url, actions[3])
                time.sleep(wait)
                run_action(base_url, actions[0])
                run_action(base_url, actions[-1])
            elif prediction == 3:# and len(boxes) == 0:
                print("backward_left")
                run_action(base_url, actions[4])
                time.sleep(wait)
                run_action(base_url, actions[0])
                run_action(base_url, actions[-1])
            elif prediction == 4:# and len(boxes) == 0:
                print("backward")
                run_action(base_url, actions[5])
                time.sleep(wait)
                run_action(base_url, actions[0])
            elif prediction == 5:# and len(boxes) == 0:
                print("backward_right")
                run_action(base_url, actions[6])
                time.sleep(wait)
                run_action(base_url, actions[0])
                run_action(base_url, actions[-1])
        else:
            run_action(base_url, actions[0])
            time.sleep(wait)'''
        time.sleep(1)

        ret, frame = cap.read()
        nFrame += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #time.sleep(10)
    #run_action(base_url, actions[1])
    #time.sleep(10)
    #run_action(base_url, actions[0])
    #run_action(base_url, actions[-1])
