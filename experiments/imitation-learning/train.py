import os
import random
import numpy as np
from preprocessing import parse_annotation
from frontend import YOLO
import json
import argparse
import pickle

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to config file')

with open(argparser.parse_args().conf) as config_buffer:    
    config = json.loads(config_buffer.read())
    
#train_imgs = parse_annotation(config['train']['train_annot_folder'], 
#                              config['train']['train_image_folder'])

train_imgs = pickle.load(open('dataset/annotation.pkl', 'rb'))

yolo = YOLO(backend = config['model']['backend'],
            input_size = config['model']['input_size'], 
            labels = config['model']['labels'], 
            max_box_per_image = config['model']['max_box_per_image'],
            anchors = config['model']['anchors'])

random.shuffle(train_imgs)

N = len(train_imgs)
train_size = int(N * config['train']['validation_split'])
print(train_size)
valid_imgs = train_imgs[train_size:]
train_imgs = train_imgs[:train_size]

yolo.train(train_imgs = train_imgs,
           valid_imgs = valid_imgs,
           nb_epochs = config['train']['nb_epochs'], 
           learning_rate = config['train']['learning_rate'], 
           batch_size = config['train']['batch_size'],
           object_scale = config['train']['object_scale'],
           no_object_scale = config['train']['no_object_scale'],
           coord_scale = config['train']['coord_scale'],
           class_scale = config['train']['class_scale'],
           saved_weights_name = config['train']['saved_weights_name'],
           debug = config['train']['debug'])
