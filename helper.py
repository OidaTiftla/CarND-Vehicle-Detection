#importing some useful packages
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

def read_img(path):
    # Read in each one by one
    img = mpimg.imread(path)
    # .png images are scaled 0 to 1 by mpimg and
    # .jpg are scaled 0 to 255
    if path.endswith('.png'):
        img = (img * 255).astype(np.uint8)
    return img

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

# Define a function that takes an image, a list of bounding boxes,
# and optional color tuple and line thickness as inputs
# then draws boxes in that color on the output
# Add bounding boxes in this format, these are just example coordinates.
# bboxes = [((280, 500), (380, 580)), ((490, 510), (550, 570)), ((860, 520), (1130, 680))]
def draw_bounding_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # make a copy of the image
    draw_img = np.copy(img)
    # draw each bounding box on your image copy using cv2.rectangle()
    for bbox in bboxes:
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    # return the image copy with boxes drawn
    return draw_img # Change this line to return image copy with boxes
