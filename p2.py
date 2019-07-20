"""
  FileName     [ p2.py ]
  PackageName  [ DLCV_HW1 ]
  Synopsis     [ Problem 2 Solution of the HW1 ]

  Problem 2
  Images : 40 people's head image, 10 grey images for each person
  Format : Open with RGB mode

  Problem 2-1
  Use PCA to reduce dimension and plot the mean and first 4 
  eigenvector for first person 

  Problem 2-2
  Reconstruct the human face by the reduced dimension vector

  Problem 2-3
  Use k-NN to classify the face.
"""

import argparse
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

import cv2
import utils

verbose = False
draw_graph = False

def pcaPattern(X_Train, mean, n: int, index: int):
    """
    Plot the image with reduced dimension

    Parameters
    ---------
    X_Train : np.ndarray
        the image

    mean : np.ndarray
        the mean vector of the training set

    n : int

    index : int

    Return
    ---------
    newX_train : 

    mse : float

    pca : sklearn.decomposition.PCA
        the instance of PCA after fitting the training dataset
    """
    pca = PCA(n_components=n, copy=True, whiten=False)
    newX_Train = pca.fit_transform(X_Train)

    reconstruct_X = (np.dot(newX_Train, pca.components_) + mean).astype(int)
    plt.subplot(2, 2, index)
    imgplot = plt.imshow(reconstruct_X[0].reshape(56, 46, 3))

    mse = sklearn.metrics.mean_squared_error(X_Train[0], reconstruct_X[0])
    
    return newX_Train, mse, pca

def kNNPattern(X_Train, X_Test, Y_Train, Y_Test, k: int):
    """
    Parameters
    ---------
    k : int
        the hyperparameter of kNN algorithm

    Return
    ---------
    trainScore : float
        the accuracy of k-NN applying in training dataset

    testScore : float
        the accuracy of k-NN applying in testing dataset
    """
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_Train, Y_Train)

    trainScore = neigh.score(X_Train, Y_Train)
    testScore  = neigh.score(X_Test, Y_Test)

    return trainScore, testScore

def main():
    # Read images
    imagesName = os.listdir("p2_data")
    images = {}
    for person in range(1, 41):  
        images[person] = {}
    
    for name in imagesName:
        personNumber = int(name[: name.index("_")])
        personCounter = int(name[name.index("_") + 1 : name.index(".")])
        
        image = cv2.imread(os.path.join("p2_data", name))
        images[personNumber][personCounter] = image

    X_Train = [images[person][counter] for counter in range(1, 7) for person in range(1, 41)]
    Y_Train = [person for counter in range(1, 7) for person in range(1, 41)]
    X_Test  = [images[person][counter] for counter in range(7, 11) for person in range(1, 41)]
    Y_Test  = [person for counter in range(7, 11) for person in range(1, 41)]
    
    # Solved as numpy element
    X_Train = np.array(X_Train)
    X_Test  = np.array(X_Test)
    Y_Train = np.array(Y_Train)
    Y_Test  = np.array(Y_Test)
    
    if verbose:
        print("The shape of X_Train: {}".format(X_Train.shape))
        print("The shape of X_Test:  {}".format(X_Test.shape))
        print("The shape of Y_Train: {}".format(Y_Train.shape))
        print("The shape of Y_Test:  {}".format(Y_Test.shape))

    # Mean face
    mean_X_Train = np.mean(X_Train, axis=0)
    
    if draw_graph:
        imgplot = plt.imshow(mean_X_Train.astype(int).reshape(56, 46, 3))
        plt.show()

    # for counter, N in enumerate([3, 45, 140, 229], 1)
    newX_Train_1, mse_1, pca_1 = pcaPattern(X_Train, mean_X_Train, 3, 1)
    newX_Train_2, mse_2, pca_2 = pcaPattern(X_Train, mean_X_Train, 45, 2)
    newX_Train_3, mse_3, pca_3 = pcaPattern(X_Train, mean_X_Train, 140, 3)
    newX_Train_4, mse_4, pca_4 = pcaPattern(X_Train, mean_X_Train, 229, 4)

    # plot first 4 pca components
    for counter, vector in enumerate(pca_4.components_[0:4], 1):
        plt.subplot(2, 2, counter)
        vector = (vector - min(vector)) / (max(vector) - min(vector)) * 255               
        
        plt.imshow(vector.reshape(56, 46, 3).astype(int))
    
    if draw_graph:
        plt.show()

    # ---------------------------------- #
    # Problem 2-4                        #
    # - Apply k-NN classifier            #
    # - Choosing parameter: k and N      #
    # ---------------------------------- #
    for k in (1, 3, 5):
        for N, X_Train in enumerate([newX_Train_1, newX_Train_2, newX_Train_3], 1):
            if N == 1:
                pca_X_Test = pca_1.transform(X_Test)
            if N == 2:
                pca_X_Test = pca_2.transform(X_Test)
            if N == 3:
                pca_X_Test = pca_3.transform(X_Test)
            
            trainScore, testScore = kNNPattern(X_Train, pca_X_Test, Y_Train, Y_Test, k)
            print("K={}-N={}:{} ,{}".format(k, N, trainScore, testScore))

if __name__ == "__main__":
    main()
