"""
  FileName     [ utils.py ]
  PackageName  [ DLCV_HW1 ]
  Synopsis     [ utility function of HW1 ]

  Problem 2:
  - read the grey images (With RGB mode)
  - PCA, plot the mean and first 4 eigenvector
  - Reconstruct the human face by the lower dimension vector
  - Using the k-NN strategy to classify the face.
"""
import os

from PIL import Image


def is_image_file(filename):
    return any(filename.lower().endswith(extension) for extension in ['.png', '.jpg', '.bmp', '.mat'])

def load_all_image(path):
    return [os.path.join(path, x) for x in os.listdir(path) if is_image_file(x)]

def combine_photo(img1, img2, output, color="RGB"):
    """ 
    Accept 2 images in the array, combine them with the horizontal direction 
    
    Parameters
    ---------
    img1 : np.ndarray

    img2 : np.ndarray

    output : str
    
    color : str

    """
    size = img1.shape[:-1]

    if color == "RGB":
        toImage = Image.new(color, (size[0] * 2, size[1]))

    img1 = Image.open(img1)
    img2 = Image.open(img2)
    toImage.paste(img1, (0, 0))
    toImage.paste(img2, (size[0], 0))
    toImage.save(output)

    return toImage