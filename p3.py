"""
  FileName     [ p3.py ]
  PackageName  [ DLCV_HW1 ]
  Synopsis     [ Problem 3 Solution of the HW1 ]

  Library:
  * scikit-learn    (version: 0.20.3)
  * scipy           (version: 1.2.1)
  * numpy           (version: 1.15.4)
  * matplotlib      (version: )
  * cv2             (version: )
"""

import numpy as np
import cv2
import os
import random
import sklearn
# from matplotlib import pyplot

# Data import
categories = os.listdir("p3_data/")
X_Train = np.array([])
Y_Train = np.array([])
X_Test  = np.array([])
Y_Test  = np.array([])
numCluster = 15
maxIteration = 5000
numNearsetNeighbor = 5

def readImages():
    for category in categories:
        directory   = "p3_data/{}".format(category)
        imageNames  = [ "{}/{}".format(directory, name) for name in os.listdir(directory) ] 

        buffer = np.empty((64, 64, 3))
        for name in imageNames: 
            buffer = np.append(buffer, cv2.imread(name), axis=0)

        X_Train = np.append(X_Train, buffer[:375], axis=0)
        X_Test  = np.append(X_Test, buffer[375:], axis=0)
        
        if category == "banana":
            Y_Train = np.append(Y_Train, np.ones(375))
        elif category == "fountain":
            Y_Train = np.append(Y_Train, 2 * np.ones(375))
        elif category == "reef":
            Y_Train = np.append(Y_Train, 3 * np.ones(375))
        elif category == "tractor":
            Y_Train = np.append(Y_Train, 4 * np.ones(375))

        if category == "banana":
            Y_Test = np.append(Y_Test, np.ones(125))
        elif category == "fountain":
            Y_Test = np.append(Y_Test, 2 * np.ones(125))
        elif category == "reef":
            Y_Test = np.append(Y_Test, 3 * np.ones(125))
        elif category == "tractor":
            Y_Test = np.append(Y_Test, 4 * np.ones(125))

    return X_Train, Y_Train, X_Test, Y_Test

def showImageSet():
    for dataset in [X_Train, Y_Train, X_Test, Y_Test]:
        print("Shape: {}".format(dataset.shape()))

def randomPickPlot(X_Train, category=4):
    for i in range(0, 4):
        randomIndex = random.choices(range(0, 375), k=4)

# Divide the image into patches
# Flatten the vector
def image2cols(image, patch_size, stride, flatten=True):
    if len(image.shape) == 2:   # Grey image
        imhigh, imwidth = image.shape
    if len(image.shape) == 3:   # RGB image
        imhigh, imwidth, imch = image.shape

    range_y = np.arange(0, imhigh - patch_size[0], stride)
    range_x = np.arange(0, imwidth - patch_size[1], stride)
    
    if range_y[-1] != imhigh - patch_size[0]:
        range_y = np.append(range_y, imhigh - patch_size[0])
    if range_x[-1] != imwidth - patch_size[1]:
        range_x = np.append(range_x, imwidth - patch_size[1])
    
    size = len(range_y) * len(range_x)
    if len(image.shape) == 2:   # Grey image
        res = np.zeros((size, patch_size[0], patch_size[1]))
    if len(image.shape) == 3:   # RGB image
        res = np.zeros((size, patch_size[0], patch_size[1], imch))
    
    index = 0
    for y in range_y:
        for x in range_x:
            patch = image[y : y+patch_size[0], x : x+patch_size[1]]
            res[index] = patch
            index = index + 1
    
    return res 

# Flatten the patches into 1D array
# X_train_patches, X_test_patches = np.Flatten()
# Check size (24000, 768), (8000, 768)

# PCA and plot with matplotlib

# Bag-of-words
# Soft-max strategy

def main():
    images = readImages()
    
    # Images -> image patches
    for category in images:
        for key in images[category].keys():
            values = np.array([])

            for value in images[category][key]:
                value.append(image2cols(value, (16, 16), 16, flatten=False))

            # Total sets
            if key == "X_Train":
                X_Train.append(values)
            if key == "X_Test":
                X_Test.append(values)

            # Categories sets
            images[category][key] = values

    # Flatten the patches into 1D array
    for patch in X_Train:
        patch.flatten()

    for patch in X_Test:
        patch.flatten()

    showImageSet()

    # k-Means Cluster
    kMeans = sklearn.cluster.KMeans(n_clusters=numCluster)
    kMeans.fit(X_Train)
    
    # PCA
    pca = sklearn.decomposition.PCA(n_components=3)
    pca.fit(X_Train)
    pca.transform(X_Train)
    pca.transform(X_Test)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)

    # Plot with matplotlib 3D

    # Bag-of-Words with softmax 

    # K-nearest neighbors
    neighbor = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3)
    neighbor.fit(X_Train, Y_Train)


if __name__ == "__main__":
    main()