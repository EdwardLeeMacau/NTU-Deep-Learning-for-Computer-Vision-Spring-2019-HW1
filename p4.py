"""
  FileName     [ p4.py ]
  PackageName  [ DLCV_HW1 ]
  Synopsis     [ Problem 4 Solution of the HW1 ]

  Problem 4:
  - use opencv to apply gaussian filter
  - calculate the 1D conv array, apply to the image
  - calculate the images conv's magnitude
"""

from PIL import Image
import cv2
import numpy as np
import os

verbose = False
show_graph = False
draw_graph = False

def combine_photo(arr, output):
    toImage = Image.new('RGB', (1024, 512))
    img1 = Image.open(arr[0])
    img2 = Image.open(arr[1])
    toImage.paste(img1, (0, 0))
    toImage.paste(img2, (512, 0))
    toImage.save(output)

def main():
    raw_image = cv2.imread("lena.png")
    if verbose: print("Raw_image.shape: {}".format(raw_image.shape))

    # Gaussian Filter
    gaussianKernal = (3, 3)
    sigma  = 1 / (2 * np.log(2))
    gaussianBlur = cv2.GaussianBlur(raw_image, gaussianKernal, sigmaX=sigma, sigmaY=sigma)

    if show_graph:
        cv2.imshow("Gaussina Blur", gaussianBlur)
        cv2.waitKey(0)
        cv2.destroyAllWindows()    

    # Export the image as the file
    if not os.path.exists("p4_answer"): os.mkdir("p4_answer")
    cv2.imwrite(os.path.join("p4_answer", "lena_gaussian.png"), gaussianBlur)

    combine_photo(["lena.png", "p4_answer/lena_gaussian.png"], "p4_answer/merged.png")

    # Self define kernal
    kernalX = np.array([0.5, 0, -0.5])
    kernalY = np.array([0.5, 0, -0.5])
    
    gaussian_image = cv2.imread("p4_answer/lena_gaussian.png")

    for image in [raw_image, gaussian_image]:
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
                # Remove the first one and last one element
                I_x = np.convolve(row, kernalX).astype(int)[1:-1]
                if index == 1:
                    image_Blue_Conv_X.append(I_x)
                elif index == 2:
                    image_Green_Conv_X.append(I_x)
                elif index == 3:
                    image_Red_Conv_X.append(I_x)

            channel = channel.transpose()
            for row in channel:
                # Remove the first one and last one element
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

        if verbose:
            print("image_Blue_Conv.shape: {}".format(image_Blue_Conv_X.shape))
            print("image_Blue_Conv.shape: {}".format(image_Green_Conv_X.shape))
            print("image_Blue_Conv.shape: {}".format(image_Red_Conv_X.shape))

        # 1D convolution
        image_Conv_X = cv2.merge([image_Blue_Conv_X, image_Green_Conv_X, image_Red_Conv_X])
        image_Conv_Y = cv2.merge([image_Blue_Conv_Y, image_Green_Conv_Y, image_Red_Conv_Y])        

        if verbose:
            print("image_Conv_X.shape: {}".format(image_Conv_X.shape))
            print("image_Conv_Y.shape: {}".format(image_Conv_Y.shape))

        # 2D(Magnitude) Convolution
        image_Conv_M = np.sqrt(image_Conv_X * image_Conv_X + image_Conv_Y * image_Conv_Y)
        
        if image.all() == raw_image.all():
            cv2.imwrite("p4_answer/conv_M.png", image_Conv_M)
            cv2.imwrite("p4_answer/conv_X.png", image_Conv_X)
            cv2.imwrite("p4_answer/conv_Y.png", image_Conv_Y)
            
            image_Conv_X = cv2.merge([image_Blue_Conv_X, image_Green_Conv_X, image_Red_Conv_X]) * 5 + 100
            image_Conv_Y = cv2.merge([image_Blue_Conv_Y, image_Green_Conv_Y, image_Red_Conv_Y]) * 5 + 100

            cv2.imwrite("p4_answer/conv_X_Shift.png", image_Conv_X)
            cv2.imwrite("p4_answer/conv_Y_Shift.png", image_Conv_Y)
            
            combine_photo(["p4_answer/conv_X.png", "p4_answer/conv_Y.png"], "p4_answer/Conv_XY.png")
            combine_photo(["p4_answer/conv_X_Shift.png", "p4_answer/conv_Y_Shift.png"], "p4_answer/Conv_XY_Shift.png")

        if image.all() == gaussian_image.all():
            cv2.imwrite("p4_answer/conv_M_Gaussian.png", image_Conv_M)
            cv2.imwrite("p4_answer/conv_X_Gaussian.png", image_Conv_X)
            cv2.imwrite("p4_answer/conv_Y_Gaussian.png", image_Conv_Y)
            
            image_Conv_X = cv2.merge([image_Blue_Conv_X, image_Green_Conv_X, image_Red_Conv_X]) * 5 + 100
            image_Conv_Y = cv2.merge([image_Blue_Conv_Y, image_Green_Conv_Y, image_Red_Conv_Y]) * 5 + 100

            cv2.imwrite("p4_answer/conv_X_Shift_Gaussian.png", image_Conv_X)
            cv2.imwrite("p4_answer/conv_Y_Shift_Gaussian.png", image_Conv_Y)

            combine_photo(["p4_answer/conv_X_Gaussian.png", "p4_answer/conv_Y_Gaussian.png"], "p4_answer/Conv_XY_Gaussian.png")
            combine_photo(["p4_answer/conv_X_Shift_Gaussian.png", "p4_answer/conv_Y_Shift_Gaussian.png"], "p4_answer/Conv_XY_Shift_Gaussian.png")

if __name__ == "__main__":
    main()