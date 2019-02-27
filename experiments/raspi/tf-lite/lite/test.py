import tensorflow as tf
import numpy as np
import cv2

dir_path = '/home/santiago/project/self-driving-car/tf-lite/'
tflite_model_file = dir_path/'model/inception_v4_299_quant.tflite'
img_path = 'elephant.jpg'

interpreter = tf.contrib.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

im = np.asarray(cv2.imread(img_path))[:,:,::-1]
im = cv2.resize(im, (299, 299))

interpreter.set_tensor(input_index, im.reshape(-1, 299, 299, 3))
interpreter.invoke()
predictions = interpreter.get_tensor(output_index)

print(predictions)
