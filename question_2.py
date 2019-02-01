#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 07:53:00 2019

@author: dohvakiin
"""

from keras.models import Model
from keras.applications.vgg16 import VGG16
import cv2
import numpy as np
from save_and_load_dataset import save_data, load_data
#from question_1 import map_labels, inv_map_labels, predict_label, compute_precision, 

def image_reshape(img_list, req_shape):
    n_imgs = len(img_list)
    img_arr = np.array([cv2.resize(x, req_shape) for x in img_list])
    assert img_arr.shape[0] == n_imgs and img_arr.shape[1] == req_shape[0] and img_arr.shape[2] == req_shape[1] and img_arr.shape[3] == 3 
    return img_arr

net = VGG16()
extractor = Model(inputs=net.inputs, outputs=net.get_layer('fc2').output)

train_imgs, train_labels, test_imgs, test_labels = load_data('Datasets.pkl')
reshaped_train_imgs = image_reshape(train_imgs, (224,224))
reshaped_test_imgs = image_reshape(test_imgs, (224,224))
train_feature_vectors = extractor.predict(reshaped_train_imgs)
test_feature_vectors = extractor.predict(reshaped_test_imgs)

dataset = [train_feature_vectors, train_labels, test_feature_vectors, test_labels]
filename = 'VGG16_features.pkl'
save_data(dataset, filename)

