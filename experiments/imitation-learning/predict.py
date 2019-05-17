import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import Vehicle
import json

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to pretrained weights')

argparser.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format)')


def _main(args):
    config_path  = args.conf
    weights_path = args.weights
    image_path   = args.input

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    model = Vehicle(backend          = config['model']['backend'],
                 input_size          = config['model']['input_size'], 
                 labels              = config['model']['labels'], 
                 max_box_per_image   = config['model']['max_box_per_image'],
                 anchors             = config['model']['anchors'])

    model.load_weights(weights_path)

    if 'http://' in image_path:
        while True:
            cap = cv2.VideoCapture(image_path)
            ret, image = cap.read()
            nout = model.predict(image)
            return config['model']['actions'][np.argmax(nout)[1]]

    else:
        image = cv2.imread(image_path)
        nout = model.predict(image)
        return config['model']['actions'][np.argmax(nout)[1]]

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
