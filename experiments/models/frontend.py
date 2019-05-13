from keras.models import Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from utils import decode_netout, compute_overlap, compute_ap
from keras.applications.mobilenet import MobileNet
from keras.layers.merge import Concatenate
from keras.optimizers import SGD, Adam, RMSprop
from keras.losses import categorical_crossentropy
from preprocessing import BatchGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from backend import TinyYoloFeature, MobileNetFeature, SqueezeNetFeature

import tensorflow as tf
import numpy as np
import os
import cv2

class Vehicle(object):
    def __init__(self, backend,
                 input_size, 
                 labels,
                 actions,
                 max_box_per_image=50,
                 anchors=[0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]):

        self.input_size = input_size        
        self.labels = list(labels)
        self.actions = list(actions)
        self.nb_act = len(self.actions)
        self.nb_class = len(self.labels)
        self.nb_box = len(anchors)//2
        self.class_wt = np.ones(self.nb_class, dtype='float32')
        self.anchors = anchors

        self.max_box_per_image = max_box_per_image

        ##########################
        # Make the model
        ##########################

        # make the feature extractor layers
        input_image     = Input(shape=(self.input_size, self.input_size, 3))
        self.true_boxes = Input(shape=(1, 1, 1, max_box_per_image , 4))  

        if backend == 'SqueezeNet':
            self.feature_extractor = SqueezeNetFeature(self.input_size)        
        elif backend == 'MobileNet':
            self.feature_extractor = MobileNetFeature(self.input_size)
        elif backend == 'Tiny Yolo':
            self.feature_extractor = TinyYoloFeature(self.input_size)
        else:
            raise Exception('Architecture not supported! Only support Full Yolo, Tiny Yolo, MobileNet, SqueezeNet, VGG16, ResNet50, and Inception3 at the moment!')

        print(self.feature_extractor.get_output_shape())    
        self.grid_h, self.grid_w = self.feature_extractor.get_output_shape()        
        features = self.feature_extractor.extract(input_image)            

        # make the object detection layer
        yolo = Conv2D(self.nb_box * (4 + 1 + self.nb_class), 
                        (1,1), strides=(1,1), 
                        padding='same', 
                        name='DetectionLayer', 
                        kernel_initializer='lecun_normal')(features)
        yolo = Reshape((self.grid_h, self.grid_w, self.nb_box, 4 + 1 + self.nb_class))(yolo)
        
        image_features = AveragePooling2D()(features, name='avpool')
        
        image_features = Flatten()(image_features)
        yolo = Flatten()(yolo)

        merged = Concatenate()([image_features, yolo])
        fc1 = Dense(32, activation='relu')(merged)
        output = Dense(self.nb_act, activation='softmax')
        
        self.model = Model([input_image, self.true_boxes], output)

        
        # initialize the weights of the detection layer
        # TODO initilize weights


    def load_weights(self, weight_path):
        self.model.load_weights(weight_path)

    def train(self, train_imgs,     # the list of images to train the model
                    valid_imgs,     # the list of images used to validate the model
                    nb_epochs,      # number of epoches
                    learning_rate,  # the learning rate
                    batch_size,     # the size of the batch
                    object_scale,
                    no_object_scale,
                    coord_scale,
                    class_scale,
                    saved_weights_name='best_weights.h5',
                    debug=False):     

        self.batch_size = batch_size

        self.object_scale    = object_scale
        self.no_object_scale = no_object_scale
        self.coord_scale     = coord_scale
        self.class_scale     = class_scale

        self.debug = debug

        ############################################
        # Make train and validation generators
        ############################################

        generator_config = {
            'IMAGE_H'         : self.input_size, 
            'IMAGE_W'         : self.input_size,
            'GRID_H'          : self.grid_h,  
            'GRID_W'          : self.grid_w,
            'BOX'             : self.nb_box,
            'LABELS'          : self.labels,
            'CLASS'           : len(self.labels),
            'ANCHORS'         : self.anchors,
            'BATCH_SIZE'      : self.batch_size,
            'TRUE_BOX_BUFFER' : self.max_box_per_image,
        }    

        train_generator = BatchGenerator(train_imgs, 
                                     generator_config, 
                                     norm=self.feature_extractor.normalize)
        valid_generator = BatchGenerator(valid_imgs, 
                                     generator_config, 
                                     norm=self.feature_extractor.normalize,
                                     jitter=False) 

        ############################################
        # Compile the model
        ############################################

        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(loss=categorical_crossentropy, optimizer=optimizer)

        ############################################
        # Make a few callbacks
        ############################################

        early_stop = EarlyStopping(monitor='val_loss', 
                           min_delta=0.001, 
                           patience=3, 
                           mode='min', 
                           verbose=1)
        checkpoint = ModelCheckpoint(saved_weights_name, 
                                     monitor='val_loss', 
                                     verbose=1, 
                                     save_best_only=True, 
                                     mode='min', 
                                     period=1)
        tensorboard = TensorBoard(log_dir=os.path.expanduser('~/logs/'), 
                                  histogram_freq=0, 
                                  #write_batch_performance=True,
                                  write_graph=True, 
                                  write_images=False)

        ############################################
        # Start the training process
        ############################################        

        self.model.fit_generator(generator        = train_generator, 
                                 steps_per_epoch  = len(train_generator), 
                                 epochs           = nb_epochs, 
                                 verbose          = 2 if debug else 1,
                                 validation_data  = valid_generator,
                                 validation_steps = len(valid_generator),
                                 callbacks        = [early_stop, checkpoint, tensorboard])      

        ############################################
        # Compute mAP on the validation set
        ############################################
        # average_precisions = self.evaluate(valid_generator)

    def evaluate(self, 
                 generator, 
                 iou_threshold=0.3,
                 score_threshold=0.3,
                 max_detections=100,
                 save_path=None):
        """ Evaluate a given dataset using a given model.
        code originally from https://github.com/fizyr/keras-retinanet

        # Arguments
            generator       : The generator that represents the dataset to evaluate.
            model           : The model to evaluate.
            iou_threshold   : The threshold used to consider when a detection is positive or negative.
            score_threshold : The score confidence threshold to use for detections.
            max_detections  : The maximum number of detections to use per image.
            save_path       : The path to save images with visualized detections to.
        # Returns
            A dict mapping class names to mAP scores.
        """    
        pass


    def predict(self, image):
        image_h, image_w, _ = image.shape
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = self.feature_extractor.normalize(image)

        input_image = image[:,:,::-1]
        input_image = np.expand_dims(input_image, 0)
        dummy_array = np.zeros((1,1,1,1,self.max_box_per_image,4))

        netout = self.model.predict([input_image, dummy_array])[0]
        boxes  = decode_netout(netout, self.anchors, self.nb_class)

        return boxes
