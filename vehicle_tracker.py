import numpy as np
import cv2
import math
import helper

class VehicleTracker:
    def __init__(self):
        pass

    def process(self, img, img_annotated, verbose=0):
        # Note: img is already the undistorted image

        # Get HOG features

        # Locate vehicles
        vehicles = []

        # Sanity checks & tracking (vehicle velocity vector)

        # Visualize the result
        img_annotated = self.visualize(img, img_annotated, vehicles, verbose)

        return img_annotated

    def visualize(self, img, img_annotated, vehicles, verbose):
        return img_annotated
