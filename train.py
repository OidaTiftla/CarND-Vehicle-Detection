import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="the input training images (.jpg,.png)", nargs='+')
args = parser.parse_args()

import numpy as np
import cv2
import matplotlib.pyplot as plt

def process_image(img):
    global vehicle_classifier
    return img

import os
import helper
from vehicle_classifier import VehicleClassifier

vehicle_classifier = None

for fname in args.input:
    ext = os.path.splitext(fname)[-1]
    if ext in ['.jpg', '.png']:
        # Read image
        print("Read image:", fname)
        img = helper.read_img(fname)
        vehicle_classifier = VehicleClassifier()
        img = process_image(img)
        if args.output:
            helper.write_img(img, 'output/' + os.path.basename(fname))
        if args.verbose >= 3:
            plt.imshow(img)
            plt.show()
    else:
        print("Unknown file extension:", fname)
