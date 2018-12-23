from styx_msgs.msg import TrafficLight
import cv2
from frontend import YOLO
import json
import os
import keras.backend as K
import tensorflow as tf
import time
import os

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.config_path = 'config.json'  # Loading config file for tiny-yolo processing

        # Check if weight file is exist or not, if not
        # create merge weight file from archive files inside
        # folder "yolo_weight_archive"
        if os.path.isfile('tiny_yolo_finalweight.h5'):
            self.weights_path = 'tiny_yolo_finalweight.h5'
        else:
            os.system('cat yolo_weight_archive/tiny_yolo_archivea* > tiny_yolo_finalweight.h5')
            self.weights_path = 'tiny_yolo_finalweight.h5'

        print(os.getcwd())
        with open(self.config_path) as config_buffer:
            config = json.load(config_buffer)

        # ==============================================
        #   Initialize Yolo model                      =
        # ==============================================
        self.yolo = YOLO(backend=config['model']['backend'],
                    input_size = config['model']['input_size'],
                    labels=config['model']['labels'],
                    max_box_per_image=config['model']['max_box_per_image'],
                    anchors=config['model']['anchors'])
        # Load the weight
        self.yolo.load_weights(self.weights_path)

        self.graph = tf.get_default_graph()

        print('Final yolo weight is loaded')




    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction

        with self.graph.as_default():
            t = time.time()
            boxes =  self.yolo.predict(image)
            elapsed_time = time.time() - t
            print('Prediction time %5.3f'%(elapsed_time,))

        if (boxes):
            class_label = boxes[0].get_label()
        else:
            class_label = None


        if class_label == 0:
            print('Predict red-color')
            print('Red-light, Number boxes %d' % (len(boxes)))
            return TrafficLight.RED
        elif class_label == 1:
            print('Green-light, Number boxes %d'%(len(boxes)))
            return TrafficLight.GREEN
        else:
            print('No-color detected')
            return TrafficLight.UNKNOWN