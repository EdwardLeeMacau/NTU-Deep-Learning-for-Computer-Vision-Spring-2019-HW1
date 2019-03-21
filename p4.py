"""
  FileName     [ p4.py ]
  PackageName  [ DLCV_HW1 ]
  Synopsis     [ Problem 4 Solution of the HW1 ]

  Library:
  * OpenCV (cv2)    version: 4.0.0
  * numpy           version: 

  Problem 1: Math proof
  Problem 2: Gaussian Filter
"""

import cv2
import numpy as np

image = cv2.imread("lena.png")

# Gaussian Filter
kernal = (3, 3)
sigma  = 1 / (2 * np.log(2))
gaussianBlur = cv2.GaussianBlur(image, kernal, sigmaX=sigma, sigmaY=sigma)


# Show on the window
# cv2.imshow("Gaussina Blur", gaussianBlur)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Gradient Magnitude
 

# Export the image as the file
cv2.imwrite("lena_gaussian.png", gaussianBlur)
