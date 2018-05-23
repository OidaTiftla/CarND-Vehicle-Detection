import numpy as np
import cv2

class LineTracker:
    def __init__(self):
        pass

    def process(self, img):
        # Note: img is the undistorted image
        img = self.color_and_gradient_filtering(img)
        return img

    def color_and_gradient_filtering(self, img):
        # Convert to HLS color space and separate the S channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]

        # Grayscale image
        # NOTE: we already saw that standard grayscaling lost color information for the lane lines
        # Explore gradients in other colors spaces / color channels to see what might work better
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Sobel x
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # Threshold x gradient
        thresh_min = 20
        thresh_max = 100
        binary_sobelx = np.zeros_like(scaled_sobelx)
        binary_sobelx[(scaled_sobelx >= thresh_min) & (scaled_sobelx <= thresh_max)] = 1

        # Threshold color channel
        s_thresh_min = 170
        s_thresh_max = 255
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

        # Stack each channel to view their individual contributions in green and blue respectively
        # This returns a stack of the two binary images, whose components you can see as different colors
        color_binary = np.dstack(( np.zeros_like(binary_sobelx), binary_sobelx, s_binary)) * 255

        # Combine the two binary thresholds
        combined_binary = np.zeros_like(binary_sobelx)
        combined_binary[(s_binary == 1) | (binary_sobelx == 1)] = 1

        # Plotting thresholded images
        import matplotlib.pyplot as plt
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.set_title('Stacked thresholds')
        ax1.imshow(color_binary)

        ax2.set_title('Combined S channel and gradient thresholds')
        ax2.imshow(combined_binary, cmap='gray')
        plt.show()

        # return binary filter
        return combined_binary
