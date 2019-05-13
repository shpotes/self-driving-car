from keras.models import Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
from keras.applications.mobilenet import MobileNet
from keras.applications import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
import tensorflow as tf


TINY_YOLO_BACKEND_PATH  = "models/tiny_yolo_backend.h5"   # should be hosted on a server
SQUEEZENET_BACKEND_PATH = "models/squeezenet_backend.h5"  # should be hosted on a server
MOBILENET_BACKEND_PATH  = "models/mobilenet_backend.h5"   # should be hosted on a server

class BaseFeatureExtractor(object):
    # to be defined in each subclass
    def __init__(self, input_size):
        raise NotImplementedError("error message")

    # to be defined in each subclass
    def normalize(self, image):
        raise NotImplementedError("error message")       

    def get_output_shape(self):
        return self.feature_extractor.get_output_shape_at(-1)[1:3]

    def extract(self, input_image):
        return self.feature_extractor(input_image)

class TinyYoloFeature(BaseFeatureExtractor):
    """Minimal yolov2 feature extractor see https://pjreddie.com/darknet/yolo/"""
    def __init__(self, input_size):
        input_image = Input(shape=(input_size, input_size, 3))

        # Layer 1
        x = Conv2D(16, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 2 - 5
        for i in range(0,4):
            x = Conv2D(32*(2**i), (3,3), strides=(1,1), padding='same', name='conv_' + str(i+2), use_bias=False)(x)
            x = BatchNormalization(name='norm_' + str(i+2))(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 6
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

        # Layer 7 - 8
        for i in range(0,2):
            x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_' + str(i+7), use_bias=False)(x)
            x = BatchNormalization(name='norm_' + str(i+7))(x)
            x = LeakyReLU(alpha=0.1)(x)

        self.feature_extractor = Model(input_image, x)  
        self.feature_extractor.load_weights(TINY_YOLO_BACKEND_PATH)

    def normalize(self, image):
        return image / 255.

class MobileNetFeature(BaseFeatureExtractor):
    """Wrapper for keras model implementation"""
    def __init__(self, input_size):
        input_image = Input(shape=(input_size, input_size, 3))

        mobilenet = MobileNet(input_shape=(224,224,3), include_top=False)
        mobilenet.load_weights(MOBILENET_BACKEND_PATH)

        x = mobilenet(input_image)

        self.feature_extractor = Model(input_image, x)  

    def normalize(self, image):
        image = image / 255.
        image = image - 0.5
        image = image * 2.

        return image		

class SqueezeNetFeature(BaseFeatureExtractor):
    """SqueezeNet reimplementation"""
    def __init__(self, input_size):

        # define some auxiliary variables and the fire module
        sq1x1  = "squeeze1x1"
        exp1x1 = "expand1x1"
        exp3x3 = "expand3x3"
        relu   = "relu_"

        def fire_module(x, fire_id, squeeze=16, expand=64):
            s_id = 'fire' + str(fire_id) + '/'

            x     = Conv2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
            x     = Activation('relu', name=s_id + relu + sq1x1)(x)

            left  = Conv2D(expand,  (1, 1), padding='valid', name=s_id + exp1x1)(x)
            left  = Activation('relu', name=s_id + relu + exp1x1)(left)

            right = Conv2D(expand,  (3, 3), padding='same',  name=s_id + exp3x3)(x)
            right = Activation('relu', name=s_id + relu + exp3x3)(right)

            x = concatenate([left, right], axis=3, name=s_id + 'concat')

            return x

        # define the model of SqueezeNet
        input_image = Input(shape=(input_size, input_size, 3))

        x = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(input_image)
        x = Activation('relu', name='relu_conv1')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

        x = fire_module(x, fire_id=2, squeeze=16, expand=64)
        x = fire_module(x, fire_id=3, squeeze=16, expand=64)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

        x = fire_module(x, fire_id=4, squeeze=32, expand=128)
        x = fire_module(x, fire_id=5, squeeze=32, expand=128)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

        x = fire_module(x, fire_id=6, squeeze=48, expand=192)
        x = fire_module(x, fire_id=7, squeeze=48, expand=192)
        x = fire_module(x, fire_id=8, squeeze=64, expand=256)
        x = fire_module(x, fire_id=9, squeeze=64, expand=256)

        self.feature_extractor = Model(input_image, x)  
        self.feature_extractor.load_weights(SQUEEZENET_BACKEND_PATH)

    def normalize(self, image):
        image = image[..., ::-1]
        image = image.astype('float')

        image[..., 0] -= 103.939
        image[..., 1] -= 116.779
        image[..., 2] -= 123.68

        return image