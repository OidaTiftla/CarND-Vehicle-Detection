import numpy as np
import cv2
import math
from scipy.ndimage.measurements import label
import helper
from vehicle_classifier import VehicleClassifier

class VehicleTracker:
    def __init__(self, classifier):
        self.classifier = classifier
        self.heatmap_over_multiple_frames = None

    def process(self, img, img_annotated, verbose=0):
        # Note: img is already the undistorted image

        hot_windows = self.classifier.search_bounding_boxes(img, verbose)
        if verbose >= 5:
            # display windows
            import matplotlib.pyplot as plt
            hot_windows_img = helper.draw_bounding_boxes(img, hot_windows, (255, 0, 0))
            plt.imshow(hot_windows_img)
            plt.show()

        # heat map
        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        heat = self.add_heat(heat, hot_windows)
        heat = np.clip(heat, 0, 5)
        if verbose >= 4:
            # display windows
            import matplotlib.pyplot as plt
            plt.imshow(heat)
            plt.show()
        # track over multiple frames
        if self.heatmap_over_multiple_frames is None:
            self.heatmap_over_multiple_frames = heat
        else:
            self.heatmap_over_multiple_frames *= 0.7
            self.heatmap_over_multiple_frames += heat
        self.heatmap_over_multiple_frames = np.clip(self.heatmap_over_multiple_frames, 0, 15)
        # threshold heat map
        heat = self.apply_threshold(self.heatmap_over_multiple_frames.copy(), 3)
        # label pixels which belong to the same cars
        labels = label(heat)
        # create bounding boxes around the identified labels
        bounding_boxes = self.get_bounding_boxes_for_labels(labels)
        # estimate position in follwoing frames

        # Locate vehicles
        vehicles = bounding_boxes

        # Sanity checks & tracking (vehicle velocity vector)

        # Visualize the result
        img_annotated = self.visualize(img, img_annotated, self.heatmap_over_multiple_frames, vehicles, verbose)

        return img_annotated

    def add_heat(self, heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap

    def apply_threshold(self, heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap

    def get_bounding_boxes_for_labels(self, labels):
        bounding_boxes = []
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # append to list
            bounding_boxes.append(bbox)
        # return all bounding boxes
        return bounding_boxes

    def visualize(self, img, img_annotated, heatmap, vehicles, verbose):
        img_annotated = helper.draw_bounding_boxes(img_annotated, vehicles)

        if verbose >= 2:
            # color heatmap
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap('jet')
            rgba_heatmap = cmap(heatmap / 15.)
            rgb_heatmap = np.delete(rgba_heatmap, 3, 2)
            rgb_heatmap = (rgb_heatmap * 255).astype(np.uint8)
            img_annotated = helper.weighted_img(rgb_heatmap, img_annotated)
        return img_annotated
