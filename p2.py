"""
  FileName     [ p2.py ]
  PackageName  [ DLCV_HW1 ]
  Synopsis     [ Problem 2 Solution of the HW1 ]

  Problem 1:
  - read the grey images
  - PCA, plot the mean and first 4 eigenvector
"""

import numpy as np
import sklearn
import cv2
import os
# from matplotlib import pyplot

imageNames = os.listdir("p2_data/")
X_Train = np.array([])
Y_Train = np.array([])
X_Test  = np.array([])
Y_Test  = np.array([])

def readImages():
    for name in imageNames:
        directory   = "p2_data/{}".format(name)
        
        buffer = np.empty((1, 56, 46))
        for name in imageNames: 
            buffer = np.append(buffer, cv2.imread(directory), axis=0)

        X_Train = np.append(X_Train, buffer[:6], axis=0)
        X_Test  = np.append(X_Test, buffer[6:], axis=0)
        
    return X_Train, Y_Train, X_Test, Y_Test

def showImageSet():
    for dataset in [X_Train, Y_Train, X_Test, Y_Test]:
        print("Shape: {}".format(dataset.shape()))

def main():
    X_Train, X_Test, Y_Train, Y_Test = readImages()
    showImageSet()

    # PCA
    # pca = sklearn.decomposition.PCA(n_components=3)
    # pca.fit(X_Train)
    # pca.transform(X_Train)
    # pca.transform(X_Test)
    # print(pca.explained_variance_ratio_)
    # print(pca.singular_values_)

    # Plot with matplotlib photo

if __name__ == "__main__":
    main()