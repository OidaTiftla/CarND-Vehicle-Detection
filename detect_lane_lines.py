import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="the input video (.mp4,.jpg,.png)", nargs='+')
parser.add_argument("--calib", help="the calibration file for the camera (.p)", default='camera.p')
# parser.add_argument("-v", "--verbose", help="show each image", action='store_true')
args = parser.parse_args()

print("Load calibration from:", args.calib)
from camera import Camera
cam = Camera.from_file(args.calib)

import numpy as np
import cv2
import matplotlib.pyplot as plt

def process_img(img):
    pass

def process_video(fname):
    # Import everything needed to edit/save/watch video clips
    from moviepy.editor import VideoFileClip

import os
import helper

for fname in args.input:
    ext = os.path.splitext(fname)[-1]
    if ext in ['.jpg', '.png']:
        # Read image
        print("Read image:", fname)
        img = helper.read_img(fname)
        process_img(img)
    elif ext in ['.mp4']:
        # Read video
        process_video(fname)
    else:
        print("Unknown file extension:", fname)
