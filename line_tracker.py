import numpy as np
import cv2
import math
import helper

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients of the last n fits of the line
        self.recent_fitted = []
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # # difference in fit coefficients between last and new fits
        # self.diffs = np.array([0,0,0], dtype='float')
        # # x values for detected line pixels
        # self.allx = None
        # # y values for detected line pixels
        # self.ally = None

class LineTracker:
    def __init__(self):
        self.left_line = Line()
        self.right_line = Line()

    def process(self, img, verbose=0):
        # Note: img is the undistorted image
        img_processed = self.color_and_gradient_filtering(img, verbose)
        img_processed, M, Minv = self.perspective_transform(img_processed, verbose)
        left_fit, right_fit, left_fitx, right_fitx, ploty = self.locate_lane_lines(img_processed, verbose)
        middlex_car = img.shape[1] / 2
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 3. / 100 # meters per pixel in y dimension
        xm_per_pix = 3.7 / 470 # meters per pixel in x dimension
        left_fit_scaled, right_fit_scaled, ploty_scaled, middlex_car_scaled = self.scale(left_fit, right_fit, ploty, middlex_car, mx=xm_per_pix, my=ym_per_pix)
        left_radius_of_curvature, right_radius_of_curvature = self.measure_curvature(left_fit_scaled, right_fit_scaled, ploty_scaled)
        offset, width, left_dir, right_dir = self.measure_lane_parameters(left_fit_scaled, right_fit_scaled, ploty_scaled, middlex_car_scaled, verbose)
        self.sanity_check(left_fit, right_fit, left_fit_scaled, right_fit_scaled, left_fitx, right_fitx, ploty, left_radius_of_curvature, right_radius_of_curvature, offset, width, left_dir, right_dir)
        img = self.visualize(img, M, Minv, left_fit, right_fit, ploty, left_radius_of_curvature, right_radius_of_curvature, offset, width, verbose)
        return img

    def color_and_gradient_filtering(self, img, verbose=0):
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

        if verbose >= 3:
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

    def perspective_transform(self, img, verbose=0):
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

        if verbose >= 4:
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

        if verbose >= 4:
            # Visualize undistortion
            import matplotlib.pyplot as plt
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
            ax1.set_title('Source image')
            ax1.imshow(img)
            ax2.set_title('Warped image')
            ax2.imshow(warped)
            plt.show()

        # return warped
        return warped, M, Minv

    def locate_lane_lines(self, img, verbose=0):
        if self.left_line.detected == False or self.right_line.detected == False:
            left_fit, right_fit, left_fitx, right_fitx, ploty = self.locate_lane_lines_histogram_search(img, verbose)
        else:
            # Skip the sliding windows step once you know where the lines are
            left_fit, right_fit, left_fitx, right_fitx, ploty = self.locate_lane_lines_based_on_last_search(img, self.left_line.best_fit, self.right_line.best_fit, verbose)

        return left_fit, right_fit, left_fitx, right_fitx, ploty

    def locate_lane_lines_histogram_search(self, img, verbose=0):
        # Assuming the imput image is a warped binary image
        # Take a histogram of the bottom half of the image
        histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
        if verbose >= 3:
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
            if verbose >= 3:
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

        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        # Generate x and y values for plotting
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        if verbose >= 3:
            import matplotlib.pyplot as plt

            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
            plt.imshow(out_img)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.xlim(0, img.shape[1])
            plt.ylim(img.shape[0], 0)
            plt.show()

        return left_fit, right_fit, left_fitx, right_fitx, ploty

    def locate_lane_lines_based_on_last_search(self, img, left_fit, right_fit, verbose=0):
        # Now you know where the lines are you have a fit! In the next frame of video you don't need to do a blind search again, but instead you can just search in a margin around the previous line position like this:

        # Assume you now have a new warped binary image
        # from the next frame of video (also called "img")
        # It's now much easier to find line pixels!
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = \
            ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin))
            & (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))

        right_lane_inds = \
            ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin))
            & (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        # Generate x and y values for plotting
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        if verbose >= 3:
            import matplotlib.pyplot as plt

            # Create an image to draw on and an image to show the selection window
            out_img = helper.ensure_color(img * 255)
            window_img = np.zeros_like(out_img)
            # Color in left and right line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
            plt.imshow(result)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.xlim(0, img.shape[1])
            plt.ylim(img.shape[0], 0)
            plt.show()

        return left_fit, right_fit, left_fitx, right_fitx, ploty

    def scale(self, left_fit, right_fit, ploty, middlex_car, mx, my):
        scaling = [mx / (my ** 2), (mx / my), mx]
        left_fit_scaled = left_fit * scaling
        right_fit_scaled = right_fit * scaling
        ploty_scaled = ploty * my
        middlex_car_scaled = middlex_car * mx
        return left_fit_scaled, right_fit_scaled, ploty_scaled, middlex_car_scaled

    def measure_curvature(self, left_fit, right_fit, ploty):
        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)
        left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
        right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

        return left_curverad, right_curverad

    def measure_lane_parameters(self, left_fit, right_fit, ploty, middlex_car, verbose=0):
        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)
        leftx = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
        rightx = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]
        middlex = (leftx + rightx) / 2
        offset = middlex - middlex_car
        width = abs(rightx - leftx)

        # direction of line
        left_slope = 2 * left_fit[0] * y_eval + left_fit[1]
        right_slope = 2 * right_fit[0] * y_eval + right_fit[1]
        left_dir = math.atan(left_slope)
        right_dir = math.atan(right_slope)

        return offset, width, left_dir, right_dir

    def sanity_check(self, left_fit, right_fit, left_fit_scaled, right_fit_scaled, left_fitx, right_fitx, ploty, left_radius_of_curvature, right_radius_of_curvature, offset, width, left_dir, right_dir):
        all_checks_passed = True

        # Checking that they have similar curvature
        left_steering_angle = math.asin(1 / left_radius_of_curvature)
        right_steering_angle = math.asin(1 / right_radius_of_curvature)
        max_diff_degree = 5
        max_diff_rad = max_diff_degree / 180. * math.pi
        if abs(left_steering_angle - right_steering_angle) > max_diff_rad:
            print("steering angle difference to large")
            all_checks_passed = False

        # Checking that they are separated by approximately the right distance horizontally
        if width >  4.7 or width < 3.15:
            print("lane width not matched:", width)
            all_checks_passed = False

        # Checking that they are roughly parallel
        max_diff_degree = 15
        max_diff_rad = max_diff_degree / 180. * math.pi
        if abs(left_dir - right_dir) > max_diff_rad:
            print("direction of the two lines do not match")
            all_checks_passed = False

        self.left_line.detected = all_checks_passed
        self.right_line.detected = all_checks_passed
        if all_checks_passed:
            # lane is detected
            max_history = 5
            self.left_line.recent_xfitted.append(left_fitx)
            self.right_line.recent_xfitted.append(right_fitx)
            if len(self.left_line.recent_xfitted) > max_history:
                self.left_line.recent_xfitted.pop(0)
            if len(self.right_line.recent_xfitted) > max_history:
                self.right_line.recent_xfitted.pop(0)

            self.left_line.bestx = np.average(np.array(self.left_line.recent_xfitted), axis=0)
            self.right_line.bestx = np.average(np.array(self.right_line.recent_xfitted), axis=0)

            self.left_line.recent_fitted.append(left_fit)
            self.right_line.recent_fitted.append(right_fit)
            if len(self.left_line.recent_fitted) > max_history:
                self.left_line.recent_fitted.pop(0)
            if len(self.right_line.recent_fitted) > max_history:
                self.right_line.recent_fitted.pop(0)

            self.left_line.best_fit = np.average(np.array(self.left_line.recent_fitted), axis=0)
            self.right_line.best_fit = np.average(np.array(self.right_line.recent_fitted), axis=0)

            self.left_line.current_fit = left_fit
            self.right_line.current_fit = right_fit

            self.left_line.radius_of_curvature = left_radius_of_curvature
            self.right_line.radius_of_curvature = right_radius_of_curvature

            self.left_line.line_base_pos = width / 2. - offset
            self.right_line.line_base_pos = width / 2. + offset

    def visualize(self, img, M, Minv, left_fit, right_fit, ploty, left_curverad, right_curverad, offset, width, verbose=0):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(img[:,:,0]).astype(np.uint8)
        color_warp = helper.ensure_color(warp_zero)

        # Recast the x and y points into usable format for cv2.fillPoly()
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
        # Combine the result with the original image
        img = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

        if verbose >= 1:
            # add some infos
            cv2.putText(img, "Curvature: {:.2f}".format((left_curverad + right_curverad) / 2), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            cv2.putText(img, "Offset: {:.2f}".format(offset), (50, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            cv2.putText(img, "Width: {:.2f}".format(width), (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

        return img
