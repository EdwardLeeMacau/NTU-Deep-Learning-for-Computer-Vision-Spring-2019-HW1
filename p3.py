"""
  FileName     [ p3.py ]
  PackageName  [ DLCV_HW1 ]
  Synopsis     [ Problem 3 Solution of the HW1 ]

  Problem 3:
  - read the images and chop as shape=(16*16)
  - use k-Means clustering to seperate the training set into C-sets
  - pca to apply the dimension reduction
  - change the words as C-dimension vector
  - k-NN detection
"""

import numpy as np
import cv2
import os
import random
import sklearn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
from mpl_toolkits.mplot3d import Axes3D

# Data import
numCluster = 15
maxIteration = 5000
numNearsetNeighbor = 5

# State setting
verbose = False
draw_graph = False

def chopImage(image, stride=16):
    imageBatchs = []

    for x in [0, 16, 32, 48]:
        for y in [0, 16, 32, 48]:
            imageBatchs.append(image[x:x+stride, y:y+stride])

    return imageBatchs

def readImages():
    # Training sets setting
    X_Train = []
    X_Test  = []
    Y_Train = []
    Y_Test  = []

    categories = os.listdir("p3_data")
    images = {}

    category_index = 1
    for number, name in enumerate(categories, 1):  
        images[name] = []
        imagesName = os.listdir("p3_data/{}".format(name))

        for number in imagesName:
            image = cv2.imread(os.path.join("p3_data", name, number))
            images[name].append(image)

        X_Train += images[name][:375]
        X_Test  += images[name][375:]
        Y_Train.append(number * np.ones(375))
        Y_Test.append(number * np.ones(125))
    
        category_index += 1

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
        image_patches = chopImage(image, stride=16)
        X_Train_patches.append(image_patches)

    for image in X_Test:
        image_patches = []

        for x in [0, 16, 32, 48]:
            for y in [0, 16, 32, 48]:
                image_patches.append(image[x:x+16, y:y+16])

        X_Test_patches.append(image_patches)

    # Change X as numpy element, Y as one hot encoding vector.
    X_Train_patches = np.array(X_Train_patches).reshape((-1, 16, 16, 3))
    Y_Train_patches = np.append(np.append(np.ones(6000), 2*np.ones(6000)), np.append(3*np.ones(6000), 4*np.ones(6000))) 
    X_Test_patches  = np.array(X_Test_patches).reshape((-1, 16, 16, 3))
    Y_Test_patches  = np.append(np.append(np.ones(2000), 2*np.ones(2000)), np.append(3*np.ones(2000), 4*np.ones(2000)))
    
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

    if verbose:
        print(centroids_pca.shape)

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

    # Bag-of-Words with softmax 
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


    # Histogram making
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
    if verbose: print("Bow_Test.shape: {}".format(BoW_Test.shape))

    neighbor = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)
    neighbor.fit(BoW, Y_Train)
    score = neighbor.score(BoW_Test, Y_Test)
    print("Bag-of-Words testing score: {}".format(score))

if __name__ == "__main__":
    main()