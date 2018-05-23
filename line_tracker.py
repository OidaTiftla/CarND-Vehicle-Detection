import numpy as np
import cv2
import helper

class LineTracker:
    def __init__(self):
        pass

    def process(self, img, verbose=False):
        # Note: img is the undistorted image
        img = self.color_and_gradient_filtering(img, verbose)
        img = self.perspective_transform(img, verbose)
        left_fit, right_fit, ploty = self.locate_lane_lines(img, verbose)
        left_curverad, right_curverad = self.measure_curvature(left_fit, right_fit, ploty, verbose)
        middlex_img = img.shape[1] / 2
        offset = self.measure_position_offset_from_middle(left_fit, right_fit, ploty, middlex_img, verbose)
        return img

    def color_and_gradient_filtering(self, img, verbose=False):
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

        if verbose:
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

    def perspective_transform(self, img, verbose=False):
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

        if verbose:
            # Source image points
            import matplotlib.pyplot as plt
            plt.imshow(img)
            plt.plot(src[0,0], src[0,1], '.') # top right
            plt.plot(src[1,0], src[1,1], '.') # bottom right
            plt.plot(src[2,0], src[2,1], '.') # bottom left
            plt.plot(src[3,0], src[3,1], '.') # top left
            plt.show()

        # Get perspective transform
        # Compute the perspective transform, M
        M = cv2.getPerspectiveTransform(src, dst)
        # Could compute the inverse also by swapping the input parameters
        Minv = cv2.getPerspectiveTransform(dst, src)
        # Create warped image - uses linear interpolation
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

        if verbose:
            # Visualize undistortion
            import matplotlib.pyplot as plt
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
            ax1.set_title('Source image')
            ax1.imshow(img)
            ax2.set_title('Warped image')
            ax2.imshow(warped)
            plt.show()

        # return warped
        return warped

    def locate_lane_lines(self, img, verbose=False):
        # Assuming the imput image is a warped binary image
        # Take a histogram of the bottom half of the image
        histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
        if verbose:
            # Create an output image to draw on and visualize the result
            out_img = helper.ensure_color(img * 255)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(img.shape[0] // nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            if verbose:
                # Draw the windows on the visualization image
                cv2.rectangle(out_img,
                    (win_xleft_low, win_y_low),
                    (win_xleft_high, win_y_high),
                    (0, 255, 0), 2)
                cv2.rectangle(out_img,
                    (win_xright_low, win_y_low),
                    (win_xright_high, win_y_high),
                    (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_xleft_low)
                & (nonzerox < win_xleft_high)
                ).nonzero()[0]
            good_right_inds = (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_xright_low)
                & (nonzerox < win_xright_high)
                ).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        if verbose:
            # Generate x and y values for plotting
            import matplotlib.pyplot as plt
            ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
            plt.imshow(out_img)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.xlim(0, img.shape[1])
            plt.ylim(img.shape[0], 0)
            plt.show()

        return left_fit, right_fit, ploty

    def measure_curvature(self, left_fit, right_fit, ploty, verbose=False):
        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)
        left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
        right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

        return left_curverad, right_curverad

    def measure_position_offset_from_middle(self, left_fit, right_fit, ploty, middlex_img, verbose=False):
        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)
        leftx = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
        rightx = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]
        middlex = (leftx + rightx) / 2
        offset = middlex - middlex_img

        return offset
