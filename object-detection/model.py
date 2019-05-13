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

feature_extractor = Model(input_image, x)  
feature_extractor.load_weights(SQUEEZENET_BACKEND_PATH)

def normalize(self, image):
    image = image[..., ::-1]
    image = image.astype('float')
    
    image[..., 0] -= 103.939
    image[..., 1] -= 116.779
    image[..., 2] -= 123.68
    
    return image    
