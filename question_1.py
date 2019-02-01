#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 18:37:50 2019

@author: dohvakiin
"""

""" import all necessary libraries """
import os
import pickle
from save_and_load_dataset import save_data, load_data
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cdist

""" function to extract the sift descriptors from images """
def extract_sift_features(img, n_octave_layers=None, sigma=None):
    if n_octave_layers is not None and sigma is None:
        sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers=n_octave_layers)
    elif n_octave_layers is not None and sigma is not None:
        sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers=n_octave_layers, sigma=sigma)
    elif n_octave_layers is None and sigma is not None:
        sift = cv2.xfeatures2d.SIFT_create(sigma=sigma)
    else:
        sift = cv2.xfeatures2d.SIFT_create()
    keypts, descriptors = sift.detectAndCompute(img, None)
    return descriptors

""" mapping and inverse mapping of labels """
def map_labels(labels):
    mapped_labels = []
    for label in labels:
        if label == 'bikes':
            code = 1
        if label == 'airplanes':
            code = 2
        if label == 'cars':
            code = 3
        if label == 'faces':
            code = 4
        mapped_labels.append(code)
    return np.array(mapped_labels)

def inv_map_labels(vectors):
    labels = []
    for vector in vectors:
        if vector==1:
            label = 'bikes'
        if vector==2:
            label = 'airplanes'
        if vector==3:
            label = 'cars'
        if vector==4:
            label = 'faces'
        labels.append(label)
    return labels

""" function to compute the feature vectors from the given clustering method """
def compute_feature_vecs(data, classifier, size):
    feature_vecs = []
    for img in data:
        centroid_assignments = classifier.predict(img)
        img_vec = np.array([np.sum(centroid_assignments==i) for i in range(size)])
        feature_vecs.append(img_vec)
    assert len(feature_vecs) == len(data)
    return feature_vecs

""" function to return visual bag of words using k-means clustering on training descriptors """
def visual_bogs(train_data, test_data, vocabulary_size=10, compute_fvs=True):
    train_mat = np.vstack(train_data)
    kmeans = KMeans(n_clusters=vocabulary_size)
    kmeans.fit(train_mat)
    centroids = kmeans.cluster_centers_
    mean_distortion = sum(np.min(cdist(train_mat, centroids, 'euclidean'), axis=1))/train_mat.shape[0]
    if compute_fvs:
        train_feature_vectors = compute_feature_vecs(train_data, kmeans, vocabulary_size)
        test_feature_vectors = compute_feature_vecs(test_data, kmeans, vocabulary_size)
        return centroids, mean_distortion, train_feature_vectors, test_feature_vectors
    else:
        return centroids, mean_distortion

""" compute k nearest neighbors classifier on the training and testing feature vectors """
def predict_label(train_data, train_labels, test_data, test_labels, k_val=5):
    train_vecs = map_labels(train_labels)
    neigh = KNeighborsClassifier(n_neighbors=k_val)
    neigh.fit(np.array(train_data), train_vecs)
    #predicted_test_labels = np.array([neigh.predict(img) for img in test_data])
    predicted_test_labels = neigh.predict(np.array(test_data))
    predicted_test_labels = inv_map_labels(predicted_test_labels)
    confusion_mat, classwise_prec, avg_prec = compute_precision(test_labels, predicted_test_labels)
    return predicted_test_labels, confusion_mat, classwise_prec, avg_prec

""" compute the classwise precision and the confusion matrix """
def compute_precision(test_labels, predicted_test_labels):
    test_labels = np.array(test_labels)
    predicted_test_labels = np.array(predicted_test_labels)
    labels = np.unique(test_labels)
    confusion_mat = np.array([[np.sum((test_labels==i)*(predicted_test_labels==j)) for j in labels] for i in labels])
    precision = {}
    avg_precision = 0
    total_labels = len(test_labels)
    for i in range(len(labels)):
        if confusion_mat[i, i] == 0:
            precision[labels[i]] = 0
        else:
            precision[labels[i]] = confusion_mat[i, i]/np.sum(confusion_mat[:, i])
        avg_precision = avg_precision + sum(test_labels==labels[i])*precision[labels[i]]/total_labels
    return confusion_mat, precision, avg_precision

""" retrieve top 4 matching images """
def retrieve_top_k(training_data, query_fv, k):
    training_data = np.array(training_data)
    query_fv = np.array(query_fv).reshape((1, -1))
    indices = np.argsort(cdist(query_fv, training_data, 'euclidean'))
    return indices[0,:k]

""" compute SIFT matching between two images based on Brute Force matcher """
def compute_descriptor_matching(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(desc1,desc2,k=2)
    matchesMask = [[0,0] for i in range(len(matches))]
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = 0)
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    return img3

""" load data and obtain the descriptors of train and test images - done"""
train_imgs, train_labels, test_imgs, test_labels = load_data('Datasets.pkl')
train_descriptors = [extract_sift_features(img) for img in train_imgs]
test_descriptors = [extract_sift_features(img) for img in test_imgs]

""" save the descriptors and bag of words into a file - done"""
centroids, mean_distortion, train_feature_vectors, test_feature_vectors = visual_bogs(train_descriptors, test_descriptors)
dataset = [train_feature_vectors, train_labels, test_feature_vectors, test_labels, centroids, mean_distortion]
filename = 'Default.pkl'
save_data(dataset, filename)

""" compute the mean distortions for a range of vocabulary sizes - tuning the optimal vocabulary size - done"""
k_vals = range(10, 310, 50)
distortions = []
for k in k_vals:
    centroids, mean_distortions = visual_bogs(train_descriptors, test_descriptors, vocabulary_size=k, compute_fvs=False)
    distortions.append(mean_distortions)
filename = 'Default_Distortions.pkl'
save_data(distortions, filename)

""" save the optimal dataset for future use - done"""
centroids, mean_distortion, train_feature_vectors, test_feature_vectors = visual_bogs(train_descriptors, test_descriptors, vocabulary_size=50)
dataset = [train_feature_vectors, train_labels, test_feature_vectors, test_labels, centroids, mean_distortion]
filename = 'Optimal.pkl'
save_data(dataset, filename)

""" try a lot of other settings for different sift parameter values given the optimal vocabulary size - done"""
""" tuning number of layers at default variance - done"""
n_layers = [3, 4, 5]
sigma_vals = [2, 4, 8]
k_opt = 50
distortions = []
for k in n_layers:
    print("Running for k = %d"%(k))
    train_descriptors = [extract_sift_features(img, n_octave_layers=k) for img in train_imgs]
    test_descriptors = [extract_sift_features(img, n_octave_layers=k) for img in test_imgs]
    centroids, mean_distortions = visual_bogs(train_descriptors, test_descriptors, vocabulary_size=k_opt, compute_fvs=False)
    distortions.append(mean_distortions)
    print("k=%d, distortion=%0.2f"%(k,mean_distortions))
filename = 'Layerwise_Distortions.pkl'
save_data(distortions, filename)

""" tuning the variance of gaussian masks at default number of layers - done"""
distortions = []
for val in sigma_vals:
    print("Running for sigma = %d"%(val))
    train_descriptors = [extract_sift_features(img, sigma=val) for img in train_imgs]
    test_descriptors = [extract_sift_features(img, sigma=val) for img in test_imgs]
    centroids, mean_distortions = visual_bogs(train_descriptors, test_descriptors, vocabulary_size=k_opt, compute_fvs=False)
    distortions.append(mean_distortions)
    print("sigma=%d, distortion=%0.2f"%(val,mean_distortions))
filename = 'Sigwise_Distortions.pkl'
save_data(distortions, filename)

""" load the optimal data and tune the knn on it - done"""
train_feature_vectors, train_labels, test_feature_vectors, test_labels, centroids, mean_distortion = load_data('Optimal.pkl')
k_neighbors = list(range(2, 42, 2))
dataset = {}
precisions = []
filename = 'Selection_of_k_for_knn.pkl'
for k in k_neighbors:
    predicted_test_labels, confusion_mat, classwise_prec, avg_prec = predict_label(train_feature_vectors, train_labels, test_feature_vectors, test_labels, k_val=k)
    dataset[k] = [predicted_test_labels, confusion_mat, classwise_prec, avg_prec]
    precisions.append(avg_prec)
save_data([dataset, precisions, k_neighbors], filename)

""" compare knn with optimal k on default and optimal datasets - done"""
k_opt = 12
train_feature_vectors, train_labels, test_feature_vectors, test_labels, centroids, mean_distortion = load_data('Default.pkl')
_, default_confusion_mat, default_classwise_prec, default_avg_prec = predict_label(train_feature_vectors, train_labels, test_feature_vectors, test_labels, k_val=k_opt)
optimum_confusion_mat = dataset[k_opt][1]
optimum_classwise_prec = dataset[k_opt][2]
optimum_avg_prec = dataset[k_opt][3]

""" plots """
""" elbow plot for optimal number of clusters or vocabulary size - done"""
k_vals, distortions = load_data('Default_Distortions.pkl')
fig=plt.figure(figsize=(40, 16), dpi=80, facecolor='w', edgecolor='k')
#plt.figure()
plt.plot(k_vals, distortions, 'ko-', lw=2)
plt.xlabel('Vocabulary size')
plt.ylabel('Average cluster distortion')
plt.title('Average cluster distortion vs. Vocabulary size')
plt.show()

""" elbow plot for number of layers per octave in SIFT - done"""
k_vals, distortions = load_data('Layerwise_Distortions.pkl')
fig=plt.figure(figsize=(40, 16), dpi=80, facecolor='w', edgecolor='k')
#plt.figure()
plt.plot(k_vals, distortions, 'ko-', lw=2)
plt.xlabel('Number of layers per octave')
plt.ylabel('Average cluster distortion')
plt.title('Average cluster distortion vs. Number of layers per octave')
plt.show()

""" elbow plot for variance of gaussian filter in SIFT - done"""
sig_vals, distortions = load_data('Sigwise_Distortions.pkl')
fig=plt.figure(figsize=(40, 16), dpi=80, facecolor='w', edgecolor='k')
#plt.figure()
plt.plot(sig_vals, distortions, 'ko-', lw=2)
plt.xlabel('$\sigma$')
plt.ylabel('Average cluster distortion')
plt.title('Average cluster distortion vs. $\sigma$')
plt.show()

""" average precision vs. no. of nearest neighbors for SIFT - done"""
fig=plt.figure(figsize=(40, 16), dpi=80, facecolor='w', edgecolor='k')
plt.plot(k_neighbors, precisions, 'ko-', lw=2)
plt.xlabel('Number of nearest neighbors')
plt.ylabel('Average classification precision across classes')
plt.title('Average Classification Precision vs No. of nearest neighbors')
plt.show()

""" confusion matrices for default and optimal datasets with optimal knn - done"""
fig=plt.figure(figsize=(40, 16), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_subplot(121)
plt.title('Vocabulary size = 10')
ax = fig.add_subplot(122)
cax2 = ax.matshow(optimum_confusion_mat)
fig.colorbar(cax2)
plt.title('Vocabulary size = 50')
plt.show()

""" retrieval and matching - done"""
""" 6th image does not match """
""" 16th image matches """
""" 11th image matches """
""" 2nd image matches """
""" 9th image matches """
""" from optimum classwise precision - airplanes and bikes have no mismatches but faces and cars do have """
""" for the 2nd image, 8 out of the first 12 matches (optimum nearest neighbors) are airplanes so it is correctly classified """
""" for the 6th image, 8 out of the first 12 matches (optimum mearest neighbors) are faces so it is correctly classified """
train_feature_vectors, train_labels, test_feature_vectors, test_labels, centroids, mean_distortion = load_data('Optimal.pkl')
train_imgs, train_labels, test_imgs, test_labels = load_data('Datasets.pkl')
query_img = test_imgs[1]
query_fv = test_feature_vectors[1]
matches = retrieve_top_k(train_feature_vectors, query_fv, 4)
fig=plt.figure(figsize=(40, 16), dpi=80, facecolor='w', edgecolor='k')
#fig=plt.figure(constrained_layout=True)
#fig, ax = plt.subplots(2,4,figsize=(4,4),constrained_layout=True)
plt.subplot(243)
plt.imshow(train_imgs[matches[0]], cmap='gray')
plt.title('1st match')
plt.subplot(244)
plt.imshow(train_imgs[matches[1]], cmap='gray')
plt.title('2nd match')
plt.subplot(247)
plt.imshow(train_imgs[matches[2]], cmap='gray')
plt.title('3rd match')
plt.subplot(248)
plt.imshow(train_imgs[matches[3]], cmap='gray')
plt.title('4th match')
plt.subplot(121)
plt.imshow(query_img, cmap='gray')
plt.title('Query')
plt.show()

""" plotting SIFT matches - done """
""" same image, same class and different class - done"""
train_imgs, train_labels, test_imgs, test_labels = load_data('Datasets.pkl')
img1 = compute_descriptor_matching(test_imgs[7], train_imgs[47])
img2 = compute_descriptor_matching(test_imgs[1], train_imgs[47])
img3 = compute_descriptor_matching(test_imgs[1], test_imgs[1])
#fig=plt.figure(figsize=(40, 16), dpi=80, facecolor='w', edgecolor='k')
fig=plt.figure()
plt.imshow(img1)
plt.title('Same class')
fig=plt.figure()
plt.imshow(img2)
plt.title('Different class')
fig=plt.figure()
plt.imshow(img3)
plt.title('Same image')
plt.show()

""" VGG16 part - done"""
""" tune k for knn - done"""
train_feature_vectors, train_labels, test_feature_vectors, test_labels = load_data('VGG16_features.pkl')
k_neighbors = list(range(2, 42, 2))
dataset = {}
precisions = []
filename = 'Selection_of_k_for_knn_VGG16.pkl'
for k in k_neighbors:
    predicted_test_labels, confusion_mat, classwise_prec, avg_prec = predict_label(train_feature_vectors, train_labels, test_feature_vectors, test_labels, k_val=k)
    dataset[k] = [predicted_test_labels, confusion_mat, classwise_prec, avg_prec]
    precisions.append(avg_prec)
save_data([dataset, precisions, k_neighbors], filename)

""" average precision vs. no. of nearest neighbors for VGG16 - done"""
#fig=plt.figure(figsize=(40, 16), dpi=80, facecolor='w', edgecolor='k')
fig=plt.figure()
plt.plot(k_neighbors, precisions, 'ko-', lw=2)
plt.xlabel('Number of nearest neighbors')
plt.ylabel('Average classification precision across classes')
plt.title('Average Classification Precision vs No. of nearest neighbors')
plt.show()
