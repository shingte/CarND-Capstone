#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np
from utils import draw_boxes
from frontend import YOLO
import json
import time

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

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


def _main_(args):
    config_path = 'config.json'
    weights_path = 'tiny_yolo_trafficred3.h5'
    image_path = args.input

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    ###############################
    #   Make the model
    ###############################

    yolo = YOLO(backend=config['model']['backend'],
                input_size=config['model']['input_size'],
                labels=config['model']['labels'],
                max_box_per_image=config['model']['max_box_per_image'],
                anchors=config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################

    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes
    ###############################


    image = cv2.imread(image_path)

    for item in range(10):
        start_time = time.time()
        boxes = yolo.predict(image)
        duration  = time.time() - start_time
        print('Prediction time %5.2f'%(duration))


    image = draw_boxes(image, boxes, config['model']['labels'])

    print(len(boxes), 'boxes are found')

    cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)


