#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 18:27:58 2019

@author: dohvakiin
"""

""" import all necessary libraries """
import cv2
import numpy as np
import os
import pickle

""" function to load data - modify this"""
def load_data(filename):
    if filename is 'Datasets.pkl':
        with open(filename, 'rb') as f:
            train_imgs, train_labels, test_imgs, test_labels = pickle.load(f)
        f.close()
        return train_imgs, train_labels, test_imgs, test_labels
    elif filename is 'Default.pkl':
        with open(filename, 'rb') as f:
            train_feature_vectors, train_labels, test_feature_vectors, test_labels, centroids, mean_distortion = pickle.load(f)
        f.close()
        return train_feature_vectors, train_labels, test_feature_vectors, test_labels, centroids, mean_distortion
    elif filename is 'Optimal.pkl':
        with open(filename, 'rb') as f:
            train_feature_vectors, train_labels, test_feature_vectors, test_labels, centroids, mean_distortion = pickle.load(f)
        f.close()
        return train_feature_vectors, train_labels, test_feature_vectors, test_labels, centroids, mean_distortion
    elif filename is 'Default_Distortions.pkl':
        with open(filename, 'rb') as f:
            distortions = pickle.load(f)
        f.close()
        return [10,60,110,160,210,260], distortions
    elif filename is 'Layerwise_Distortions.pkl':
        with open(filename, 'rb') as f:
            distortions = pickle.load(f)
        f.close()
        return [3,4,5], distortions
    elif filename is 'Sigwise_Distortions.pkl':
        with open(filename, 'rb') as f:
            distortions = pickle.load(f)
        f.close()
        return [2,4,8], distortions

""" copy this line """
train_imgs, train_labels, test_imgs, test_labels = load_data()