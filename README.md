# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

```bash
python train.py --input-cars vehicles/ --input-non-cars non-vehicles/ -o vehicle_classifier.p
```

In this project, your goal is to write a software pipeline to detect vehicles in a video (start with the test_video.mp4 and later implement on full project_video.mp4), but the main output or product we want you to create is a detailed writeup of the project.  Check out the [writeup template](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup.

Creating a great writeup:
---
A great writeup should include the rubric points as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :).

You can submit your writeup in markdown or use another method and submit a pdf instead.

The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.

Some example images for testing your pipeline on single frames are located in the `test_images` folder.  To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include them in your writeup for the project by describing what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.

**As an optional challenge** Once you have a working pipeline for vehicle detection, add in your lane-finding algorithm from the last project to do simultaneous lane-finding and vehicle detection!

**If you're feeling ambitious** (also totally optional though), don't stop there!  We encourage you to go out and take video of your own, and show us how you would implement this project on a new video!

# Advanced Lane Finding Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

In this project, the goal is to write a software pipeline to identify the lane boundaries in a video.

## The Project

The goals / steps of this project are the following:

- Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
- Apply a distortion correction to raw images.
- Use color transforms, gradients, etc., to create a thresholded binary image.
- Apply a perspective transform to rectify binary image ("birds-eye view").
- Detect lane pixels and fit to find the lane boundary.
- Determine the curvature of the lane and vehicle position with respect to center.
- Warp the detected lane boundaries back onto the original image.
- Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing the pipeline on single frames.

Examples of the output from each stage of the pipeline are saved in the folder called `output_images`.
The video called `project_video.mp4` is the video your pipeline should work well on.

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.
The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious (again, totally optional though), don't stop there!
We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!

### Changes for resubmission 1

1. I accidentally drawed the current fit, not the averaged best fit into the output image/video (this was changed only for testing and I forgot to change it back, before submitting)
2. Compare best fit with current fit (offset of the car within the lane; the lane width)
3. Do a weighted average for the best fit (using a exponential function)
4. Filter by L channel of HLS image and adjust S channel filter
5. Draw annotated image into output video

## The rubric points are considered individually and described how each point is addressed in the implementation

### Camera Calibration

The code for this step is contained in `calibrate_camera.py` and `camera.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world.
Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.
Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.
`imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.
I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![calibration1.jpg](camera_cal/calibration1.jpg)
![undistort_output.png](examples/undistort_output.png)

After calibrating, the distortion coefficients are stored in a pickle file, to be used afterwards by the `detect_lane_lines.py` file.

You can rerun the calibration by executing this line in the shell:

```bash
python calibrate_camera.py --images camera_cal/* --output camera.p
```

### Pipeline (single images)

#### An example of a distortion-corrected image

![distortion_corrected_image.png](examples/distortion_corrected_image.png)

#### Color transforms and gradients to create a thresholded binary image

The lane detection and tracking is done in the `line_tracker.py` file.
For each step of the pipeline, there is one function, which handles this step.
The function `process(img)` handles the combination of all steps and calculats some transformations needed by those steps.

The code for the first step is contained in the function called `color_and_gradient_filtering(img)` and is a combination of color and gradient thresholds, which generate a binary image.
Here's an example of my output for this step.

![binary_combo.png](examples/binary_combo.png)

#### Perspective transform

The code for the perspective transform is contained in the function called `get_perspective_transform_parameters(img_size)`, which calculates the two matrices `M` and `Minv`, and in the `perspective_transform(img, M, Minv)`.

The source and destination points were hardcoded in the following manner:

```python
# Define calibration box in source (original) and destination (desired or warped) coordinates
bottom_width = 0.6
top_width = 0.075
top = 0.63
bottom = 0.95
# Four source coordinates
src = np.float32(
    [[int((size[0] * (1. - bottom_width)) / 2.), int(size[1] * bottom)],
        [size[0] - int((size[0] * (1. - bottom_width)) / 2.), int(size[1] * bottom)],
        [size[0] - int((size[0] * (1. - top_width)) / 2.), int(size[1] * top)],
        [int((size[0] * (1. - top_width)) / 2.), int(size[1] * top)]])
# Four desired coordinates
vertical_offset = 0.98
horizontal_offset = 0.6
dst = np.float32(
    [[int((size[0] * (1. - bottom_width * horizontal_offset)) / 2.), int(size[1] * vertical_offset)],
        [size[0] - int((size[0] * (1. - bottom_width * horizontal_offset)) / 2.), int(size[1] * vertical_offset)],
        [size[0] - int((size[0] * (1. - bottom_width * horizontal_offset)) / 2.), size[1] - int(size[1] * vertical_offset)],
        [int((size[0] * (1. - bottom_width * horizontal_offset)) / 2.), size[1] - int(size[1] * vertical_offset)]])
```

This resulted in the following source and destination points:

![projection_points.png](examples/projection_points.png)

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

#### Identify lane-line pixels and fit their positions with a polynomial

In the function `locate_lane_lines(img, ploty)` the lane lines get detected.
In the first run of this function, no lane lines were detected previously, so it calls the histogram search function called `locate_lane_lines_histogram_search(img, ploty)`.
In the next run (for example the second frame of a video) the lane lines may be detected and the function `locate_lane_lines_based_on_last_search(img, ploty, left_best_fit, right_best_fit)` gets called.
In both functions the lane lines get fit with a 2nd order polynomial kinda like this:

![histogram_search.png](examples/histogram_search.png)

#### Calculate the radius of curvature of the lane and the position of the vehicle with respect to center

This is all done during checking the lane for plausibility in the function `sanity_checks(left_fit, right_fit, ploty, middlex_car)`.

First all detected parameters get converted from pixel space into meters.
This is done in the function called `scale(left_fit, right_fit, ploty, middlex_car, mx, my)`.

Afterwards the lane is splitted vertically into 5 checkpoint lines.
A sanity check is done for each of those 5 horizontal lines.
So the function `sanity_checks(left_fit, right_fit, ploty, middlex_car)` calculates all relevant parameters (including curvature).
All calculated parameters are checked in the function `sanity_check(left_radius_of_curvature, right_radius_of_curvature, width, left_dir, right_dir)`.
If the check is passed, the lane is considered as detected.

The function `measure_curvature(left_fit_scaled, right_fit_scaled, y)` calculates the curvature at a specific `y` value.
And `measure_lane_parameters(left_fit_scaled, right_fit_scaled, middlex_car_scaled, y)` measures the vehicle position within the lane, as well as the angle of the lane lines and the width of the lane.

#### Result plotted back down onto the road such that the lane area is identified clearly

The function `visualize(img, M, Minv, left_best_fit, right_best_fit, ploty, left_radius_of_curvature, right_radius_of_curvature, offset, width)` visualizes the result by plotting the detected lane are back into the source image.
It also add some information about the current lane (curvature, width and vehicle position).

![example_output.png](examples/example_output.png)

### Pipeline (video)

#### Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](output/project_video_2018-05-31_16-36-26.mp4)

You can rerun the video by executing this line in the shell:

```bash
python detect_lane_lines.py --input project_video.mp4 -ovv
```

Where `-o` says the output should be saved into the `output`-folder and `-vv` sets the verbose level to `2` (showing some calculated road parameters and the two-down-view).
The maximum verbose level is 4, and can be defined by `-vvvv`.

### Discussion

#### Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

If the lane lines are not detected in many consecutive frames, the algorithm would currently only think, the last detected lane is the current best lane.
This would result in a frozen lane line and would have drastic impact on a real car.
A real car should allert any management system, that the lane detection is currently not working, and possibly stop the car savly.

To improve the lane detection: if the best search does not meet the specifications required in the sanity check, the algorithm would simply return `no lane found`, but there might be more than one lane line going in the right direction, but the width is to big.
For example the lane next to the lane, where the car is currently driving on.
To improve this, you may not only consider two lines and see if the meet specifications, but you may search for the 5 lines, which are the best possibilities to be a lane line, and then check all combinations of those lines, and see if one is the lane you are searching for.
