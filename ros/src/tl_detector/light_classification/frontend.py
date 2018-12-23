import os


from keras.models import Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf
import numpy as np
import cv2
from utils import decode_netout, compute_overlap, compute_ap
from keras.layers.merge import concatenate
from keras.optimizers import SGD, Adam, RMSprop
from backend import TinyYoloFeature
import keras



class YOLO(object):
    def __init__(self, backend,
                 input_size,
                 labels,
                 max_box_per_image,
                 anchors):

        self.input_size = input_size

        self.labels = list(labels)
        self.nb_class = len(self.labels)
        self.nb_box = len(anchors) // 2
        self.class_wt = np.ones(self.nb_class, dtype='float32')
        self.anchors = anchors

        self.max_box_per_image = max_box_per_image

        ##########################
        # Make the model
        ##########################

        # make the feature extractor layers
        input_image = Input(shape=(self.input_size, self.input_size, 3))
        self.true_boxes = Input(shape=(1, 1, 1, max_box_per_image, 4))

        if backend == 'Tiny Yolo':
            self.feature_extractor = TinyYoloFeature(self.input_size)
        else:
            raise Exception('Only Tiny Yolo is supported')

        print('Tiny Yolo is loaded')
        print(self.feature_extractor.get_output_shape())
        self.grid_h, self.grid_w = self.feature_extractor.get_output_shape()
        features = self.feature_extractor.extract(input_image)

        # make the object detection layer

        output = Conv2D(self.nb_box * (4 + 1 + self.nb_class),
                        (1, 1), strides=(1, 1),
                        padding='same',
                        name='DetectionLayer',
                        kernel_initializer='lecun_normal')(features)
        output = Reshape((self.grid_h, self.grid_w, self.nb_box, 4 + 1 + self.nb_class))(output)
        output = Lambda(lambda args: args[0])([output, self.true_boxes])

        self.model = Model([input_image, self.true_boxes], output)

        # initialize the weights of the detection layer
        layer = self.model.layers[-4]
        weights = layer.get_weights()

        new_kernel = np.random.normal(size=weights[0].shape) / (self.grid_h * self.grid_w)
        new_bias = np.random.normal(size=weights[1].shape) / (self.grid_h * self.grid_w)

        layer.set_weights([new_kernel, new_bias])

        # print a summary of the whole model
        self.model.summary()

        print('Yolo-weight is successfully loaded')


    def load_weights(self, weight_path):
        self.model.load_weights(weight_path)


    def predict(self, image):
        image_h, image_w, _ = image.shape
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = self.feature_extractor.normalize(image)

        input_image = image[:,:,::-1]
        input_image = np.expand_dims(input_image, 0)
        dummy_array = np.zeros((1,1,1,1,self.max_box_per_image,4))

        netout = self.model.predict([input_image, dummy_array])[0]
        boxes  = decode_netout(netout, self.anchors, self.nb_class, 0.4, 0.2)

        return boxes

