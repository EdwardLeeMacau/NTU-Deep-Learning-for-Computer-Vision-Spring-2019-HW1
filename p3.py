"""
  FileName     [ p3.py ]
  PackageName  [ DLCV_HW1 ]
  Synopsis     [ Problem 3 Solution of the HW1 ]

  Problem 3:
  Images : color images, with 4 catagories
  Format : RGB, size=(64, 64), crop as (16, 16), stride is (16, 16)

  Problem 3-1:
  Use k-Means clustering to seperate the training set into C-sets
  
  Problem 3-2:
  Use PCA to apply the dimension reduction
  
  Problem 3-3:
  Change the visual words as C-dimension vector
  
  Problem 3-4:
  k-NN detection
"""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

import cv2

# Data import
numCluster = 15
maxIteration = 5000
numNearsetNeighbor = 5

# State setting
verbose = False
draw_graph = False

def cropImage(image, size=16, stride=16):
    """
    crop the images

    Parameters
    ---------
    image : np.ndarray
        The image prepared to crop

    size : {int, tuple}
        The size of small images

    stride : {int, tuple}
        The stride for every cropping

    Return
    ---------
    smallimages : list
    """
    
    width, height = image.shape

    smallimages = []

    for x in range(0, width, stride):
        for y in range(0, height, stride):
            smallimages.append(image[x:x+size, y:y+size])

    return smallimages

def readImages():
    """
    Preparing the training data and testing data

    Return
    ---------
    (X_Train, X_Test, Y_Train, Y_Test) : tuple
        dataset for problem 3
    """
    X_Train = []
    X_Test  = []
    Y_Train = []
    Y_Test  = []

    categories = os.listdir("p3_data")
    images = {}
    
    for index, name in enumerate(categories, 1):  
        images[name] = []
        imagesName = os.listdir("./p3_data/{}".format(name))

        for number in imagesName:
            image = cv2.imread(os.path.join("./p3_data", name, number))
            images[name].append(image)

        X_Train.extend(images[name][:375])
        X_Test.extend(images[name][375:])

        Y_Train.append(index * np.ones(375))
        Y_Test.append(index * np.ones(125))

    X_Train = np.array(X_Train)
    X_Test  = np.array(X_Test)
    Y_Train = np.array(Y_Train).flatten()
    Y_Test  = np.array(Y_Test).flatten()

    return X_Train, X_Test, Y_Train, Y_Test

def main():
    # Read raw image
    X_Train, X_Test, Y_Train, Y_Test = readImages()

    X_Train_patches = []
    X_Test_patches  = []

    # Make patches
    for image in X_Train:
        image_patches = cropImage(image, stride=16)
        X_Train_patches.append(image_patches)

    for image in X_Test:
        image_patches = cropImage(image, stride=16)
        X_Test_patches.append(image_patches)

    # Change X as numpy element, Y as one hot encoding vector.
    X_Train_patches = np.array(X_Train_patches).reshape((-1, 16, 16, 3))
    Y_Train_patches = np.repeat([1, 2, 3, 4], repeats=6000)
    X_Test_patches  = np.array(X_Test_patches).reshape((-1, 16, 16, 3))
    Y_Train_patches = np.repeat([1, 2, 3, 4], repeats=6000)
    
    # Plot 3 patch in each image
    if draw_graph:
        for i in range(0, 4):
            pick = random.randint(0, 374)
            patches = X_Train_patches[ i * 375 * 16 + pick * 16 : i * 375 * 16 + pick * 16 + 16]
            
            for patch in patches[0:3]:
                cv2.imshow("patch", patch)
                cv2.waitKey(0)

    X_Train_patches = np.array(X_Train_patches).reshape((24000, 768))
    X_Test_patches  = np.array(X_Test_patches).reshape((8000, 768))

    if verbose:
        print("X_Train.shape: {}".format(X_Train_patches.shape))
        print("X_Test.shape:  {}".format(X_Test_patches.shape))
        print("Y_Train.shape: {}".format(Y_Train_patches.shape))
        print("Y_Test.shape:  {}".format(Y_Test_patches.shape))

    # k-Means Cluster
    kMeans = KMeans(n_clusters=15, max_iter=5000)
    Y_Train_patches_predict = kMeans.fit_predict(X_Train_patches)
    centroids = kMeans.cluster_centers_
    labels    = kMeans.labels_
    
    # PCA to show the points
    pca = sklearn.decomposition.PCA(n_components=3)
    X_Train_patches_pca = pca.fit_transform(X_Train_patches)
    centroids_pca       = pca.transform(centroids)

    # Choose 6 clusters to visualize
    target_Cluster = random.sample([i for i in range(0, 15)], 6)

    # Plot points
    fig = plt.figure()
    axis = fig.add_subplot(111, projection='3d')
    
    color_index = 0
    colors = ['b', 'g', 'r', 'y', 'k', 'c']
    markers = ["$a$", "$b$", "$c$", "$d$", "$e$", "$f$"]

    for target_plot in target_Cluster:
        targets = []
        
        for element in range(0, 24000):
            if labels[element] == target_plot:
                targets.append(X_Train_patches_pca[element])

        targets = np.array(targets).transpose()
        # print("Targets.shape: {}".format(targets.shape))
        axis.scatter(targets[0], targets[1], targets[2], s=0.15, c=colors[color_index], marker=",")

        color_index += 1

    # plot centroids
    color_index = 0
    for target_plot in target_Cluster:
        center = centroids_pca[target_plot]
        axis.scatter(center[0], center[1], center[2], s=80, c=colors[color_index], marker='s')
        
        color_index += 1

    plt.show()

    # ------------------------------------------ #
    # Problem 3-3:                               #
    # - Bag-of-Words with softmax normalization  #
    # - Show the statistics with histogram       #
    # ------------------------------------------ #
    BoW = []

    for i in range(0, 1500):
        image_In_patches = X_Train_patches[16 * i: 16*i + 16]

        normsMatrix = []
        for patch in image_In_patches:
            norms = np.array([np.linalg.norm(patch - center) for center in centroids])
            norms = (1 / norms) / (np.sum(1 / norms))
            normsMatrix.append(norms)
        
        normsMatrix = np.array(normsMatrix).transpose()
        bow = np.amax(normsMatrix, axis=1)

        BoW.append(bow)

    BoW = np.array(BoW)
    print("Bow.shape: {}".format(BoW.shape))

    # Generate a histogram
    j = 0 
    while j < 4:
        for i in range(0, 1500):
            if Y_Train[i] == (j + 1):
                plt.subplot(2, 2, j + 1)
                plt.bar(range(0, 15), BoW[i])
                plt.title("Type {}".format(j + 1))
                
                break
    
        j += 1

    plt.show()

    # ------------------------------------------ #
    # Problem 3-4:                               #
    # - K-nerest neighbors classifier            #
    # ------------------------------------------ #
    # K-nearest neighbors
    BoW_Test = []

    for i in range(0, 500):
        image_In_patches = X_Test_patches[16 * i: 16*i + 16]

        normsMatrix = []
        for patch in image_In_patches:
            norms = np.array([np.linalg.norm(patch - center) for center in centroids])
            norms = (1 / norms) / (np.sum(1 / norms))
            normsMatrix.append(norms)
        
        normsMatrix = np.array(normsMatrix).transpose()
        bow = np.amax(normsMatrix, axis=1)

        BoW_Test.append(bow)

    BoW_Test = np.array(BoW_Test)
    
    if verbose: 
        print("Bow_Test.shape: {}".format(BoW_Test.shape))

    # K-NN Classifier
    neighbor = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)
    neighbor.fit(BoW, Y_Train)
    score = neighbor.score(BoW_Test, Y_Test)
    print("Bag-of-Words testing score: {}".format(score))

if __name__ == "__main__":
    main()
