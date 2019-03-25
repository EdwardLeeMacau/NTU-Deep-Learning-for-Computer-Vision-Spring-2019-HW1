import os
from PIL import Image

def is_image_file(filename):
    return any(filename.lower().endswith(extension) for extension in ['.png', '.jpg', '.bmp', '.mat'])

def load_all_image(path):
    return [os.path.join(path, x) for x in os.listdir(path) if is_image_file(x)]

def combine_photo(arr, output, color="RGB"):
    """ Accept 2 images in the array, combine them with the horizontal direction """
    size = arr[0].shape[:-1]

    if color == "RGB":
        toImage = Image.new(color, (size[0] * 2, size[1]))

    img1 = Image.open(arr[0])
    img2 = Image.open(arr[1])
    toImage.paste(img1, (0, 0))
    toImage.paste(img2, (size[0], 0))
    toImage.save(output)