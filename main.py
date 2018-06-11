import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="the input video (.mp4,.jpg,.png)", nargs='+')
parser.add_argument("--calib", help="the calibration file for the camera (.p)", default='camera.p')
parser.add_argument("--classifier", help="the classifier file for the vehicle_classifier (.p)", default='vehicle_classifier.p')
parser.add_argument("-v", "--verbose", help="level of verbosity (specify this option up to 5 times, for the most detailed output)", action='count', default=0)
parser.add_argument("-o", "--output", help="save output into output directory", action='store_true')
parser.add_argument("-s", "--samples", help="save each frame from video into test_images directory", action='store_true')
args = parser.parse_args()

print("Load calibration from:", args.calib)
from camera import Camera
cam = Camera.from_file(args.calib)

print("Load vehicle classifier from:", args.classifier)
from vehicle_classifier import VehicleClassifier
vehicle_classifier = VehicleClassifier.from_file(args.classifier)

print("Using the following parameters:")
print('classify_img_size =', vehicle_classifier.classify_img_size)
print('color_space =', vehicle_classifier.color_space)
print('spatial_size =', vehicle_classifier.spatial_size)
print('hist_bins =', vehicle_classifier.hist_bins)
print('hist_range =', vehicle_classifier.hist_range)
print('orient =', vehicle_classifier.orient)
print('pix_per_cell =', vehicle_classifier.pix_per_cell)
print('cell_per_block =', vehicle_classifier.cell_per_block)
print('hog_channels =', vehicle_classifier.hog_channels)

import numpy as np
import cv2
import matplotlib.pyplot as plt

def process_image(img):
    global cam
    global line_tracker
    global vehicle_tracker
    img = cam.undistort(img)
    if args.verbose >= 5:
        plt.imshow(img)
        plt.show()
    img_annotated = img.copy()
    img_annotated = line_tracker.process(img, img_annotated, args.verbose)
    img_annotated = vehicle_tracker.process(img, img_annotated, args.verbose)
    img_annotated = helper.ensure_color(img_annotated)
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    return img_annotated

def process_video_frame(get_frame, t):
    img = get_frame(t)
    if args.samples:
        helper.write_img(img, 'test_images/' + os.path.splitext(fname)[0] + '_' + '{:3.2f}'.format(t) + '.jpg')
    img = process_image(img)
    if args.samples:
        helper.write_img(img, 'test_images/' + os.path.splitext(fname)[0] + '_' + '{:3.2f}'.format(t) + '_result.jpg')
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
    # project_video.mp4
    # clip = VideoFileClip(fname).subclip(20, 27)
    # clip = VideoFileClip(fname).subclip(20, 22)
    # clip = VideoFileClip(fname).subclip(38, 43)
    # clip = VideoFileClip(fname).subclip(48.5, None)
    clip = VideoFileClip(fname)
    clip = clip.fl(process_video_frame) #NOTE: this function expects color images!!
    if args.output:
        clip.write_videofile(output_fname, audio=False)

import os
import helper
from line_tracker import LineTracker
from vehicle_tracker import VehicleTracker

line_tracker = None
vehicle_tracker = None

for fname in args.input:
    ext = os.path.splitext(fname)[-1]
    if ext in ['.jpg', '.png']:
        # Read image
        print("Read image:", fname)
        img = helper.read_img(fname)
        line_tracker = LineTracker()
        vehicle_tracker = VehicleTracker(vehicle_classifier)
        img = process_image(img)
        if args.output:
            helper.write_img(img, 'output/' + os.path.basename(fname))
        if args.verbose >= 3:
            plt.imshow(img)
            plt.show()
    elif ext in ['.mp4']:
        # Read video
        line_tracker = LineTracker()
        vehicle_tracker = VehicleTracker(vehicle_classifier)
        process_video(fname)
    else:
        print("Unknown file extension:", fname)
