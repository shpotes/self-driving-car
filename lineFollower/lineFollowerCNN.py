# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from imutils import paths
import numpy as np
import argparse
import random
import cv2
import os
import matplotlib

def build(width, height, depth, classes):
    # initialize the model
    model = Sequential()
    inputShape = (height, width, depth)
    # if we are using "channels first", update the input shape
    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
    # first set of CONV => RELU => POOL layers
    model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # second set of CONV => RELU => POOL layers
    model.add(Conv2D(50, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # second set of CONV => RELU => POOL layers
    model.add(Conv2D(30, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # second set of CONV => RELU => POOL layers
    model.add(Conv2D(10, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # second set of CONV => RELU => POOL layers
    model.add(Conv2D(40, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))
    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))
    # return the constructed network architecture
    return model

datasetNums = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
#datasetNums = [22]
baseDatasets = './lineDatasets/'
# dataset = './trainImages/' # provide the path where your training images are present
# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS = 15
INIT_LR = 1e-3
BS = 32

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

for dNum in datasetNums:
    # grab the image paths and randomly shuffle them
    #print(baseDatasets + 'dataset_' + str(dNum))
    txtPath = baseDatasets + 'dataset_' + str(dNum) + '/dataset.txt'
    #print('txtPath: {}'.format(txtPath))
    imagePaths = sorted(list(paths.list_images(baseDatasets + 'dataset_' + str(dNum))))
    #print(imagePaths)
    random.seed(42)
    random.shuffle(imagePaths)
    #print(imagePaths)

    # loop over the input images
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (56, 56))
        image = img_to_array(image)
        data.append(image)
        # extract the class label from the image path and update the
        # labels list

        direction = ''
        lineID = int(imagePath.split("/")[-1][:-4])
        f = open(txtPath, "r")
        for i,linea in enumerate(f):
            if i<lineID:
                continue
            else:
                direction = linea[:-1].split(" ")[1]
                break;
        #print(str(lineID) + " " + str(direction) + " " + imagePath)
        if direction == "forward_left":
            label = 0
        elif direction == "forward":
            label = 1
        elif direction == "forward_right":
            label = 2
        elif direction == "backward_left":
            label = 3
        elif direction == "backward":
            label = 4
        elif direction == "backward_right":
            label = 5
        #print(direction)
        #print(label)
        labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=6)
testY = to_categorical(testY, num_classes=6)
# initialize the model
print("[INFO] compiling model...")
model = build(width=56, height=56, depth=3, classes=6)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network...")
trainX = trainX.reshape(-1,56, 56, 3)   #Reshape for CNN -  should work!!
testX = testX.reshape(-1,56, 56, 3)
H = model.fit(trainX, trainY, batch_size=BS, validation_data=(testX, testY), epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save("anonymous_model.h5")
