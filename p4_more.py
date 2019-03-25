"""
  FileName     [ p4.py ]
  PackageName  [ DLCV_HW1 ]
  Synopsis     [ Problem 4 Solution of the HW1 ]
"""

from PIL import Image
import cv2
import numpy as np
import os

def combine_photo(arr, output):
    toImage = Image.new('RGB', (1024, 512))
    img1 = Image.open(arr[0])
    img2 = Image.open(arr[1])
    toImage.paste(img1, (0, 0))
    toImage.paste(img2, (512, 0))
    toImage.save(output)

def main():
    raw_image = cv2.imread("lena.png")
    gaussian_image = cv2.imread("p4_answer/lena_gaussian.png")

    # Self define kernal
    kernalX = np.array([0.5, 0, -0.5])
    kernalY = np.array([0.5, 0, -0.5])

    if True:
        image = raw_image
        image_Blue, image_Green, image_Red = cv2.split(image)

        image_Blue_Conv_X = []
        image_Green_Conv_X = []
        image_Red_Conv_X = []
        
        image_Blue_Conv_Y = []
        image_Green_Conv_Y = []
        image_Red_Conv_Y = []

        index = 1
        for channel in [image_Blue, image_Green, image_Red]:

            for row in channel:
                I_x = np.convolve(row, kernalX).astype(int)[1:-1]
                if index == 1:
                    image_Blue_Conv_X.append(I_x)
                elif index == 2:
                    image_Green_Conv_X.append(I_x)
                elif index == 3:
                    image_Red_Conv_X.append(I_x)

            channel = channel.transpose()
            for row in channel:
                I_y = np.convolve(row, kernalY).astype(int)[1:-1]
                if index == 1:
                    image_Blue_Conv_Y.append(I_y)
                elif index == 2:
                    image_Green_Conv_Y.append(I_y)
                elif index == 3:
                    image_Red_Conv_Y.append(I_y)
            
            index += 1

        image_Blue_Conv_X = np.array(image_Blue_Conv_X)
        image_Green_Conv_X = np.array(image_Green_Conv_X)
        image_Red_Conv_X = np.array(image_Red_Conv_X)
        
        image_Blue_Conv_Y = np.array(image_Blue_Conv_Y).transpose()
        image_Green_Conv_Y = np.array(image_Green_Conv_Y).transpose()
        image_Red_Conv_Y = np.array(image_Red_Conv_Y).transpose()

        image_Conv_X = cv2.merge([image_Blue_Conv_X, image_Green_Conv_X, image_Red_Conv_X])
        image_Conv_Y = cv2.merge([image_Blue_Conv_Y, image_Green_Conv_Y, image_Red_Conv_Y])        
        
        image_Conv_M = np.sqrt(image_Conv_X * image_Conv_X + image_Conv_Y * image_Conv_Y)

        cv2.imwrite("p4_compare/conv_M.png", image_Conv_M)

    if True:
        image = gaussian_image
        image_Blue, image_Green, image_Red = cv2.split(image)

        image_Blue_Conv_X = []
        image_Green_Conv_X = []
        image_Red_Conv_X = []
        
        image_Blue_Conv_Y = []
        image_Green_Conv_Y = []
        image_Red_Conv_Y = []

        index = 1
        for channel in [image_Blue, image_Green, image_Red]:

            for row in channel:
                I_x = np.convolve(row, kernalX).astype(int)[1:-1]
                if index == 1:
                    image_Blue_Conv_X.append(I_x)
                elif index == 2:
                    image_Green_Conv_X.append(I_x)
                elif index == 3:
                    image_Red_Conv_X.append(I_x)

            channel = channel.transpose()
            for row in channel:
                I_y = np.convolve(row, kernalY).astype(int)[1:-1]
                if index == 1:
                    image_Blue_Conv_Y.append(I_y)
                elif index == 2:
                    image_Green_Conv_Y.append(I_y)
                elif index == 3:
                    image_Red_Conv_Y.append(I_y)
            
            index += 1

        image_Blue_Conv_X = np.array(image_Blue_Conv_X)
        image_Green_Conv_X = np.array(image_Green_Conv_X)
        image_Red_Conv_X = np.array(image_Red_Conv_X)
        
        image_Blue_Conv_Y = np.array(image_Blue_Conv_Y).transpose()
        image_Green_Conv_Y = np.array(image_Green_Conv_Y).transpose()
        image_Red_Conv_Y = np.array(image_Red_Conv_Y).transpose()

        image_Conv_X = cv2.merge([image_Blue_Conv_X, image_Green_Conv_X, image_Red_Conv_X])
        image_Conv_Y = cv2.merge([image_Blue_Conv_Y, image_Green_Conv_Y, image_Red_Conv_Y])        
        
        image_Conv_M = np.sqrt(image_Conv_X * image_Conv_X + image_Conv_Y * image_Conv_Y)

        cv2.imwrite("p4_compare/conv_M_Gaussian.png", image_Conv_M)
            
    combine_photo(["p4_compare/conv_M.png", "p4_compare/conv_M_Gaussian.png"], "p4_compare/merged_M.png")

if __name__ == "__main__":
    main()