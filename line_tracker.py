import numpy as np
import cv2

class LineTracker:
    def __init__(self):
        pass

    def process(self, img):
        # Note: img is the undistorted image
        img = self.color_and_gradient_filtering(img)
        img = self.perspective_transform(img)
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

        # # Plotting thresholded images
        # import matplotlib.pyplot as plt
        # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        # ax1.set_title('Stacked thresholds')
        # ax1.imshow(color_binary)
        # ax2.set_title('Combined S channel and gradient thresholds')
        # ax2.imshow(combined_binary, cmap='gray')
        # plt.show()

        # return binary filter
        return combined_binary

    def perspective_transform(self, img):
        import matplotlib.pyplot as plt

        img_size = (img.shape[1], img.shape[0])

        # Define calibration box in source (original) and destination (desired or warped) coordinates
        bottom_width = 0.6
        top_width = 0.075
        top = 0.63
        bottom = 0.95
        # Four source coordinates
        src = np.float32(
            [[int((img_size[0] * (1. - bottom_width)) / 2.), int(img_size[1] * bottom)],
             [img_size[0] - int((img_size[0] * (1. - bottom_width)) / 2.), int(img_size[1] * bottom)],
             [img_size[0] - int((img_size[0] * (1. - top_width)) / 2.), int(img_size[1] * top)],
             [int((img_size[0] * (1. - top_width)) / 2.), int(img_size[1] * top)]])
        # Four desired coordinates
        vertical_offset = 0.98
        horizontal_offset = 0.6
        dst = np.float32(
            [[int((img_size[0] * (1. - bottom_width * horizontal_offset)) / 2.), int(img_size[1] * vertical_offset)],
             [img_size[0] - int((img_size[0] * (1. - bottom_width * horizontal_offset)) / 2.), int(img_size[1] * vertical_offset)],
             [img_size[0] - int((img_size[0] * (1. - bottom_width * horizontal_offset)) / 2.), img_size[1] - int(img_size[1] * vertical_offset)],
             [int((img_size[0] * (1. - bottom_width * horizontal_offset)) / 2.), img_size[1] - int(img_size[1] * vertical_offset)]])

        # # Source image points
        # plt.imshow(img)
        # plt.plot(src[0,0], src[0,1], '.') # top right
        # plt.plot(src[1,0], src[1,1], '.') # bottom right
        # plt.plot(src[2,0], src[2,1], '.') # bottom left
        # plt.plot(src[3,0], src[3,1], '.') # top left
        # plt.show()

        # Get perspective transform
        # Compute the perspective transform, M
        M = cv2.getPerspectiveTransform(src, dst)
        # Could compute the inverse also by swapping the input parameters
        Minv = cv2.getPerspectiveTransform(dst, src)
        # Create warped image - uses linear interpolation
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

        # # Visualize undistortion
        # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        # ax1.set_title('Source image')
        # ax1.imshow(img)
        # ax2.set_title('Warped image')
        # ax2.imshow(warped)
        # plt.show()

        # return warped
        return warped
