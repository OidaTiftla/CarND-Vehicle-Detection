#importing some useful packages
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

def read_img(path):
    return mpimg.imread(path)

def write_img(img, path):
    if len(img.shape) > 2:
        # convert RGB to BGR (BGR2RGB works just fine)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # create directories if they do not exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # save image
    cv2.imwrite(path, img)

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def hsv(img):
    """Applies the HSV transform"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Or use BGR2HSV if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def hls(img):
    """Applies the HLS transform"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # Or use BGR2HLS if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

def ensure_color(img):
    if len(img.shape) > 2:
        return img
    else:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)