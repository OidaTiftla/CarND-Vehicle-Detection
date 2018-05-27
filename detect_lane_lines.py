import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="the input video (.mp4,.jpg,.png)", nargs='+')
parser.add_argument("--calib", help="the calibration file for the camera (.p)", default='camera.p')
parser.add_argument("-v", "--verbose", help="level of verbosity (specify this option up to 4 times, for the most detailed output)", action='count', default=0)
parser.add_argument("-o", "--output", help="save output into output directory", action='store_true')
args = parser.parse_args()

print("Load calibration from:", args.calib)
from camera import Camera
cam = Camera.from_file(args.calib)

import numpy as np
import cv2
import matplotlib.pyplot as plt

def process_image(img):
    global line_tracker
    img = cam.undistort(img)
    if args.verbose >= 4:
        plt.imshow(img)
        plt.show()
    img = line_tracker.process(img, args.verbose)
    img = helper.ensure_color(img)
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    return img

def process_video(fname):
    # Import everything needed to edit/save/watch video clips
    from moviepy.editor import VideoFileClip

    # nowStr for the filenames of the outputs
    import datetime
    now = datetime.datetime.now()
    nowStr = now.strftime("%Y-%m-%d_%H-%M-%S")
    output_fname = 'output/' + os.path.splitext(fname)[0] + '_' + nowStr + '.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip = VideoFileClip(fname).subclip(0,5)
    clip = VideoFileClip(fname)
    clip = clip.fl_image(process_image) #NOTE: this function expects color images!!
    if args.output:
        clip.write_videofile(output_fname, audio=False)

import os
import helper
from line_tracker import LineTracker

line_tracker = None

for fname in args.input:
    ext = os.path.splitext(fname)[-1]
    if ext in ['.jpg', '.png']:
        # Read image
        print("Read image:", fname)
        img = helper.read_img(fname)
        line_tracker = LineTracker()
        img = process_image(img)
        if args.output:
            helper.write_img(img, 'output/' + os.path.basename(fname))
        if args.verbose >= 2:
            plt.imshow(img)
            plt.show()
    elif ext in ['.mp4']:
        # Read video
        line_tracker = LineTracker()
        process_video(fname)
    else:
        print("Unknown file extension:", fname)
