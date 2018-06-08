import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--input-cars", help="the input training images showing cars (.jpg,.png)", nargs='+')
parser.add_argument("-n", "--input-non-cars", help="the input training images showing non-cars (.jpg,.png)", nargs='+')
parser.add_argument("-o", "--output", help="the output file, where to save the trained classifier (.p)")
args = parser.parse_args()

import numpy as np
import cv2
import matplotlib.pyplot as plt

import os
import helper
import matplotlib.image as mpimg
from vehicle_classifier import VehicleClassifier, VehicleClassifierTrainer

vehicle_classifier_trainer = VehicleClassifierTrainer()

def read_images(filenames, label):
    global vehicle_classifier_trainer
    for fname in filenames:
        ext = os.path.splitext(fname)[-1]
        if ext in ['.jpg', '.png']:
            # Read image
            print("Read image:", fname)
            img = helper.read_img(fname)
            vehicle_classifier_trainer.add_training_img(img, label)
        else:
            print("Unknown file extension:", fname)

# read images
read_images(args.input_cars, 1)
read_images(args.input_non_cars, 0)

# train classifier
vehicle_classifier = vehicle_classifier_trainer.train()

# save classifier
print("Save classifier to:", args.output)
vehicle_classifier.save(args.output)
