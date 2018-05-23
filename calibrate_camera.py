import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--images", help="a list of calibration images", nargs='+')
parser.add_argument("--chessboard", help="the size of the chessboard = ?x?", default='9x6')
parser.add_argument("-o", "--output", help="the output file, where to save the calibration (.p)")
parser.add_argument("-v", "--verbose", help="show each image", action='store_true')
args = parser.parse_args()

# Parse chessboard size
splitted = args.chessboard.split('x')
chessboard_size = (int(splitted[0]), int(splitted[1]))
print("Using chessboard size:", chessboard_size)

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Arrays to store object points and image points from all the images
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (7,5,0)
objp = np.zeros((chessboard_size[0]*chessboard_size[1],3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1, 2) # x, y coordinates

for fname in args.images:
    # Read in each image
    print("Read image:", fname)
    img = mpimg.imread(fname)

    # Convert image to grayscale (mpimg.imread)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Convert image to grayscale (cv2.imread)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # If corners are found, add object points, image points
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
        if args.verbose:
            # # draw and display the corners
            img = cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
            plt.imshow(img)
            plt.show()
    else:
        print("FAILED to identify chessboard!!!")
        if args.verbose:
            plt.imshow(img)
            plt.show()

# Get calibration for that camera
print("Calibrate camera...")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape, None, None)


# Undistorts an image
# dst = cv2.undistort(img, mtx, dist, None, mtx)

print("Save calibration to:", args.output)
from camera import Camera
cam = Camera(mtx, dist)
cam.save(args.output)
