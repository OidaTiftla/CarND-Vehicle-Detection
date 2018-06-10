import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--input-cars", help="the input training images showing cars (.jpg,.png)", nargs='+')
parser.add_argument("-n", "--input-non-cars", help="the input training images showing non-cars (.jpg,.png)", nargs='+')
parser.add_argument("-o", "--output", help="the output file, where to save the trained classifier (.p)")
args = parser.parse_args()

import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

import os
import glob
import helper
import matplotlib.image as mpimg
from vehicle_classifier import VehicleClassifier, VehicleClassifierTrainer

vehicle_classifier_trainer = VehicleClassifierTrainer()

def read_images(filenames, label):
    global vehicle_classifier_trainer
    for fname in filenames:
        if os.path.isdir(fname):
            files = glob.glob(os.path.join(fname, '*'))
            read_images(files, label)
        else:
            ext = os.path.splitext(fname)[-1]
            if ext in ['.jpg', '.png']:
                # Read image
                img = helper.read_img(fname)
                vehicle_classifier_trainer.add_training_img(img, label)
            else:
                print("Unknown file extension:", fname)

# read images
print("Reading images and extracting features...")
t1 = time.time()
read_images(args.input_cars, 1)
read_images(args.input_non_cars, 0)
t2 = time.time()
print(round(t2-t1, 2), 'seconds to read images and extract features...')

# train classifier
vehicle_classifier = vehicle_classifier_trainer.train()

# save classifier
if not(args.output is None):
    print("Save classifier to:", args.output)
    vehicle_classifier.save(args.output)
