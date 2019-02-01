#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 17:28:13 2019

@author: dohvakiin
"""

""" import all necessary libraries """
import cv2
import numpy as np
import os
import pickle

""" reading files into training and testing data and generating the labels accordingly """
def group_data(parent_dirs, path):
    file_list = []
    label_list = []
    for folder in parent_dirs:
        label = folder.split("_")[0]
        files = os.listdir(path+folder)
        for file in files:
            img = cv2.imread(path+folder+'/'+file)
            file_list.append(img)
            label_list.append(label)
    return file_list, label_list   

""" function to save data to a file """
def save_data(dataset, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)
    f.close()
    return 

""" function to load data from the given file name"""
def load_data(filename):
    if filename == 'Datasets.pkl':
        with open(filename, 'rb') as f:
            train_imgs, train_labels, test_imgs, test_labels = pickle.load(f)
        f.close()
        return train_imgs, train_labels, test_imgs, test_labels
    elif filename == 'Default.pkl':
        with open(filename, 'rb') as f:
            train_feature_vectors, train_labels, test_feature_vectors, test_labels, centroids, mean_distortion = pickle.load(f)
        f.close()
        return train_feature_vectors, train_labels, test_feature_vectors, test_labels, centroids, mean_distortion
    elif filename == 'Optimal.pkl':
        with open(filename, 'rb') as f:
            train_feature_vectors, train_labels, test_feature_vectors, test_labels, centroids, mean_distortion = pickle.load(f)
        f.close()
        return train_feature_vectors, train_labels, test_feature_vectors, test_labels, centroids, mean_distortion
    elif filename == 'Default_Distortions.pkl':
        with open(filename, 'rb') as f:
            distortions = pickle.load(f)
        f.close()
        return [10,60,110,160,210,260], distortions
    elif filename == 'Layerwise_Distortions.pkl':
        with open(filename, 'rb') as f:
            distortions = pickle.load(f)
        f.close()
        return [3,4,5], distortions
    elif filename == 'Sigwise_Distortions.pkl':
        with open(filename, 'rb') as f:
            distortions = pickle.load(f)
        f.close()
        return [2,4,8], distortions
    elif filename == 'Selection_of_k_for_knn.pkl':
        with open(filename, 'rb') as f:
            dataset, precisions, k_neighbors = pickle.load(f)
        f.close()
        return dataset, precisions, k_neighbors
    elif filename == 'VGG16_features.pkl':
        with open(filename, 'rb') as f:
            train_feature_vectors, train_labels, test_feature_vectors, test_labels = pickle.load(f)
        f.close()
        return train_feature_vectors, train_labels, test_feature_vectors, test_labels
    else:
        print("Wrong file name")
        return 

dirpath = "/home/dohvakiin/Desktop/AIP ASSIGNMENTS/new data/assignment1_data/"
subdirs = [f.name for f in os.scandir(dirpath) if f.is_dir()]
train_dirs = [x for x in subdirs if 'train' in x]
test_dirs = [x for x in subdirs if 'test' in x]
train_imgs, train_labels = group_data(train_dirs, dirpath)
test_imgs, test_labels = group_data(test_dirs, dirpath)
train_labels = [train_labels[i] for i in range(len(train_labels)) if train_imgs[i] is not None]
train_imgs = [train_imgs[i] for i in range(len(train_imgs)) if train_imgs[i] is not None]
test_labels = [test_labels[i] for i in range(len(test_labels)) if test_imgs[i] is not None]
test_imgs = [test_imgs[i] for i in range(len(test_imgs)) if test_imgs[i] is not None]

""" write the data into a master file """
dataset = [train_imgs, train_labels, test_imgs, test_labels]
filename = 'Datasets.pkl'
save_data(dataset, filename)


