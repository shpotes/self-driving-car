from keras.preprocessing.image import img_to_array
from keras.models import load_model
import argparse
import io
import requests
import time
import cv2
import numpy as np
import time

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

    if(not connection_ok(base_url)):
        raise Exception('Connection failed')

    #cap = cv2.VideoCapture(base_url + '?action=stream')
    cap = cv2.VideoCapture('http://10.42.0.235:8080/?action=stream')
    #cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture('linesData2.mp4')
    ret, frame = cap.read()

    if(not ret):
        raise Exception('no image read')

    model = load_model("anonymous_model.h5")
    nFrame = 0
    wait = 0.3
    waitFrames = 40 # Number of frames to ignore
    while(ret and (nFrame<20)):
        # Capture frame-by-frame
        #cv2.imshow("frame", frame)
        #print(nFrame)
        image = frame#.array
        image = cv2.resize(image, (56, 56), interpolation = cv2.INTER_AREA)
        cv2.imshow("frame", frame)
        image = img_to_array(image)
        image = np.array(image, dtype="float") / 255.0
        image = image.reshape(-1, 56, 56, 3)


        for i in range(waitFrames):
            _,_ = cap.read()

        i_time = time.time()
        prediction = np.argmax(model.predict(image))
        print('Time predicting action: {}'.format(time.time()-i_time))
        print("prediction = {} ... action = {}".format(prediction, actions[prediction+1]))
        if prediction == 0:
            print("forward_left")
            run_action(base_url, actions[1])
            time.sleep(wait)
            run_action(base_url, actions[0])
            run_action(base_url, actions[-1])
        elif prediction == 1:
            print("forward")
            run_action(base_url, actions[2])
            time.sleep(wait)
            run_action(base_url, actions[0])
        elif prediction == 2:
            print("forward_right")
            run_action(base_url, actions[3])
            time.sleep(wait)
            run_action(base_url, actions[0])
            run_action(base_url, actions[-1])
        elif prediction == 3:
            print("backward_left")
            run_action(base_url, actions[4])
            time.sleep(wait)
            run_action(base_url, actions[0])
            run_action(base_url, actions[-1])
        elif prediction == 4:
            print("backward")
            run_action(base_url, actions[5])
            time.sleep(wait)
            run_action(base_url, actions[0])
        elif prediction == 5:
            print("backward_right")
            run_action(base_url, actions[6])
            time.sleep(wait)
            run_action(base_url, actions[0])
            run_action(base_url, actions[-1])
        time.sleep(1)

        ret, frame = cap.read()
        nFrame += 1

        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break
    #time.sleep(10)
    #run_action(base_url, actions[1])
    #time.sleep(10)
    #run_action(base_url, actions[0])
    #run_action(base_url, actions[-1])
