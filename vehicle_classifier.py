import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import time
import helper
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
# NOTE: the next import is only valid
# for scikit-learn version <= 0.17
# if you are using scikit-learn >= 0.18 then use this:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split

class VehicleClassifier:
    def __init__(self, scaler, classifier,
                classify_img_size,
                color_space,
                spatial_size,
                hist_bins,
                hist_range,
                orient,
                pix_per_cell,
                cell_per_block,
                hog_channel):
        self.scaler = scaler
        self.classifier = classifier
        self.classify_img_size = classify_img_size
        self.color_space = color_space # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.hist_range = hist_range
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel # Can be 0, 1, 2, 'GRAY' or 'ALL'

        self.windows = None

    def save(self, fname):
        data = {}
        data['scaler'] = self.scaler
        data['classifier'] = self.classifier
        data['classify_img_size'] = self.classify_img_size
        data['color_space'] = self.color_space
        data['spatial_size'] = self.spatial_size
        data['hist_bins'] = self.hist_bins
        data['hist_range'] = self.hist_range
        data['orient'] = self.orient
        data['pix_per_cell'] = self.pix_per_cell
        data['cell_per_block'] = self.cell_per_block
        data['hog_channel'] = self.hog_channel
        pickle.dump(data, open(fname, "wb"))

    @classmethod
    def from_file(cls, fname):
        data = pickle.load(open(fname, "rb"))
        return cls(
            data['scaler'],
            data['classifier'],
            data['classify_img_size'],
            data['color_space'],
            data['spatial_size'],
            data['hist_bins'],
            data['hist_range'],
            data['orient'],
            data['pix_per_cell'],
            data['cell_per_block'],
            data['hog_channel']
        )

    def classify(self, features):
        return self.classifier.predict(features)

    def search_bounding_boxes(self, img, verbose=0):
        if self.windows is None:
            # sliding windows
            windows = []
            imwidth = img.shape[1]
            imheight = img.shape[0]

            # street parameters
            far_left = (580, 460)
            far_right = (701, 460)
            near_left = (234, 700)
            near_right = (1069, 700)
            def get_width_for_y(y):
                width_far = far_right[0] - far_left[0]
                width_near = near_right[0] - near_left[0]
                delta_width = width_far - width_near
                delta_y = far_left[1] - near_left[1]
                m = float(delta_y) / delta_width
                t = far_left[1] - m * width_far
                width = (y - t) / m
                return width
            # create windows
            for y in [445, 460, 480, 510]:
                # y = 439 + ((700 - 439) / 5.) * i
                width = get_width_for_y(y)
                new_windows = self.slide_window(img,
                                x_start_stop=[np.clip(imwidth / 2. - 6 * width, 0, imwidth), np.clip(imwidth / 2. + 6 * width, 0, imwidth)],
                                # x_start_stop=[None, None],
                                y_start_stop=[np.clip(y - 0.6 * width, 0, imheight), np.clip(y + width, 0, imheight)],
                                xy_window=(width, width), xy_overlap=(0.7, 0.7))
                if verbose >= 5:
                    # display windows
                    import matplotlib.pyplot as plt
                    search_windows_img = helper.draw_bounding_boxes(img, new_windows, (0, 255, 255))
                    search_windows_img = helper.draw_bounding_boxes(search_windows_img, [new_windows[0], new_windows[-1]], (255, 0, 0))
                    plt.imshow(search_windows_img)
                    plt.show()
                windows += new_windows
            # filter windows, which are not squares
            def aspect_ratio(w):
                (x1, y1), (x2, y2) = w
                width = x2 - x1
                height = y2 - y1
                return width / height
            windows = [w for w in windows if abs(aspect_ratio(w) - 1) < 0.05]

            # display windows
            if verbose >= 1:
                print('Searching windows:', len(windows))
            if verbose >= 4:
                import matplotlib.pyplot as plt
                search_windows_img = helper.draw_bounding_boxes(img, windows, (0, 255, 255))
                plt.imshow(search_windows_img)
                plt.show()
            self.windows = windows

        # subsampling HOG features
        # classify
        hot_windows = self.search_windows(img, self.windows)

        return hot_windows

    # Define a function to extract features from a list of images
    # Have this function call bin_spatial() and color_hist()
    def extract_features(self, img):
        # Convert image to new color space (if specified)
        if self.color_space == 'RGB':
            feature_image = np.copy(img)
            #feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif self.color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif self.color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif self.color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif self.color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif self.color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        # color histograms
        hist_features = self.color_hist(feature_image)
        # 32x32 raw pixels in some color space
        spatial_features = self.bin_spatial(feature_image)
        # HOG features
        if self.hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(self.get_hog_features(feature_image[:,:,channel],
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        elif self.hog_channel == 'GRAY':
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hog_features = self.get_hog_features(gray,
                                    vis=False, feature_vec=True)
        else:
            hog_features = self.get_hog_features(feature_image[:,:,self.hog_channel],
                                    vis=False, feature_vec=True)
        # combine features
        features = np.concatenate([hist_features, spatial_features, hog_features])
        return features

    # Define a function that takes an image,
    def slide_window(self, img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        xy_window = np.int_(np.round(np.array(xy_window)))
        xy_overlap = np.array(xy_overlap)
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None: x_start_stop[0] = 0
        if x_start_stop[1] == None: x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None: y_start_stop[0] = 0
        if y_start_stop[1] == None: y_start_stop[1] = img.shape[0]
        # Compute the span of the region to be searched
        xy_span = np.array((x_start_stop[1] - x_start_stop[0], y_start_stop[1] - y_start_stop[0]))
        # Compute the number of pixels per step in x/y
        xy_pix_per_step = np.int_(np.floor(xy_window * (1 - xy_overlap)))
        # Compute the number of windows in x/y
        xy_steps = np.int_(np.floor((xy_span - xy_window) / xy_pix_per_step + 1))
        xy_steps[xy_steps < 1] = 1
        xy_pix_per_step = (xy_span - xy_window) / (xy_steps - 1.)
        xy_pix_per_step[xy_steps <= 1] = 0
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        #     Note: you could vectorize this step, but in practice
        #     you'll be considering windows one by one with your
        #     classifier, so looping makes sense
        for yi in range(xy_steps[1]):
            if xy_steps[1] <= 1:
                y = np.int(np.round((y_start_stop[1] - y_start_stop[0]) / 2. + y_start_stop[0] - xy_window[1] / 2.))
            else:
                y = np.int(np.floor(y_start_stop[0] + yi * xy_pix_per_step[1]))
            for xi in range(xy_steps[0]):
                if xy_steps[0] <= 1:
                    x = np.int(np.round((x_start_stop[1] - x_start_stop[0]) / 2. + x_start_stop[0] - xy_window[0] / 2.))
                else:
                    x = np.int(np.floor(x_start_stop[0] + xi * xy_pix_per_step[0]))
                # Calculate each window position
                w = np.array(((x, y), tuple((x, y) + xy_window)))
                w[:,0] = np.clip(w[:,0], x_start_stop[0], x_start_stop[1])
                w[:,1] = np.clip(w[:,1], y_start_stop[0], y_start_stop[1])
                w = tuple(map(tuple, w))
                # Append window position to list
                window_list.append(w)
        # Return the list of windows
        return window_list

    # Define a function you will pass an image
    # and the list of windows to be searched (output of slide_windows())
    def search_windows(self, img, windows):
        #1) Create an empty list to receive positive detection windows
        on_windows = []
        #2) Iterate over all windows in the list
        for window in windows:
            #3) Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], self.classify_img_size)
            #4) Extract features for that window using single_img_features()
            features = self.extract_features(test_img)
            #5) Scale extracted features to be fed to classifier
            test_features = self.normalize(np.array(features).reshape(1, -1))
            #6) Predict using your classifier
            prediction = self.classify(test_features)
            #7) If positive (prediction == 1) then save the window
            if prediction == 1:
                on_windows.append(window)
        #8) Return windows for positive detections
        return on_windows

    def normalize(self, features):
        # Apply the scaler to features
        return self.scaler.transform(features)

    # Define a function to compute color histogram features
    def color_hist(self, img):
        if self.hist_bins == 0 or self.hist_range[0] == self.hist_range[1]:
            return np.array([])
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=self.hist_bins, range=self.hist_range)
        channel2_hist = np.histogram(img[:,:,1], bins=self.hist_bins, range=self.hist_range)
        channel3_hist = np.histogram(img[:,:,2], bins=self.hist_bins, range=self.hist_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    # Define a function to compute color histogram features
    # Pass the color_space flag as 3-letter all caps string
    # like 'HSV' or 'LUV' etc.
    # KEEP IN MIND IF YOU DECIDE TO USE THIS FUNCTION LATER
    # IN YOUR PROJECT THAT IF YOU READ THE IMAGE WITH
    # cv2.imread() INSTEAD YOU START WITH BGR COLOR!
    def bin_spatial(self, img):
        if self.spatial_size[0] == 0 or self.spatial_size[1] == 0:
            return np.array([])
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, self.spatial_size).ravel()
        # Return the feature vector
        return features

    # Define a function to return HOG features and visualization
    # Features will always be the first element of the return
    # Image data will be returned as the second element if visualize= True
    # Otherwise there is no second return element
    def get_hog_features(self, img, vis=False, feature_vec=True):
        return hog(img,
            orientations=self.orient,
            pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
            cells_per_block=(self.cell_per_block, self.cell_per_block),
            visualise=vis,
            feature_vector=feature_vec,
            block_norm="L2-Hys",
            transform_sqrt=False)

def rotate_img(image, angle_deg):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle_deg, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

class VehicleClassifierTrainer:
    def __init__(self):
        self.files_list = []
        self.labels_list = []

        # parameters
        classify_img_size = (64, 64)
        color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        spatial_size = (0, 0) # Spatial binning dimensions
        hist_bins = 0 # Number of histogram bins
        hist_range = (0, 256) # Range of histogram
        orient = 9 # HOG orientations
        pix_per_cell = 16 # HOG pixels per cell
        cell_per_block = 2 # HOG cells per block
        hog_channels = 'ALL' # Can be 0, 1, 2, 'GRAY' or 'ALL'

        self.classifier = VehicleClassifier(None, None,
            classify_img_size,
            color_space,
            spatial_size,
            hist_bins,
            hist_range,
            orient,
            pix_per_cell,
            cell_per_block,
            hog_channels)

    def train(self):
        print('Data summary')
        print('============')
        hist = np.histogram(self.labels_list, bins=2, range=(0, 1))
        print('Total cars:', hist[0][1], 'non-cars:', hist[0][0])
        # Split up data into randomized training and test sets
        rand_state = 76 #np.random.randint(0, 100)
        files_train, files_test, labels_train, labels_test = train_test_split(
            self.files_list, self.labels_list,
            test_size=0.2, random_state=rand_state)
        hist = np.histogram(labels_train, bins=2, range=(0, 1))
        print('Train cars:', hist[0][1], 'non-cars:', hist[0][0])
        hist = np.histogram(labels_test, bins=2, range=(0, 1))
        print('Test cars:', hist[0][1], 'non-cars:', hist[0][0])

        # read images
        print("Reading images and extracting features...")
        t1 = time.time()
        # training dataset
        X_train = []
        y_train = []
        for i in range(len(files_train)):
            fname = files_train[i]
            label = labels_train[i]
            img = helper.read_img(fname)
            if img.shape[0] != self.classifier.classify_img_size[0] or img.shape[1] != self.classifier.classify_img_size[1]:
                print('input shape not', self.classifier.classify_img_size)
                img = cv2.resize(img, self.classifier.classify_img_size)
            X_train.append(self.classifier.extract_features(img))
            y_train.append(label)
            # augment training set
            X_train.append(self.classifier.extract_features(rotate_img(img, 5)))
            y_train.append(label)
            X_train.append(self.classifier.extract_features(rotate_img(img, -5)))
            y_train.append(label)
            X_train.append(self.classifier.extract_features(rotate_img(img, 10)))
            y_train.append(label)
            X_train.append(self.classifier.extract_features(rotate_img(img, -10)))
            y_train.append(label)
        X_train, y_train = shuffle(X_train, y_train)
        hist = np.histogram(y_train, bins=2, range=(0, 1))
        print('Train cars:', hist[0][1], 'non-cars:', hist[0][0], '[after augmentation]')

        # testing dataset
        X_test = []
        y_test = []
        for i in range(len(files_test)):
            fname = files_test[i]
            label = labels_test[i]
            img = helper.read_img(fname)
            if img.shape[0] != self.classifier.classify_img_size[0] or img.shape[1] != self.classifier.classify_img_size[1]:
                print('input shape not', self.classifier.classify_img_size)
                img = cv2.resize(img, self.classifier.classify_img_size)
            X_test.append(self.classifier.extract_features(img))
            y_test.append(label)
        t2 = time.time()
        print(round(t2-t1, 2), 'seconds to read images and extract features...')

        # Create an array stack, NOTE: StandardScaler() expects np.float64
        X_train = np.vstack(X_train).astype(np.float64)
        X_test = np.vstack(X_test).astype(np.float64)
        # Fit a per-column scaler
        self.classifier.scaler = StandardScaler().fit(X_train)

        X_train = self.classifier.normalize(X_train)
        X_test = self.classifier.normalize(X_test)

        # Use a linear SVC (good in speed and accuracy)
        svc = LinearSVC()
        # Check the training time for the SVC
        print('Start training SVC...')
        t1 = time.time()
        # train classifier
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t1, 2), 'seconds to train SVC...')
        # Check the score of the SVC
        print('Train Accuracy of SVC = ', round(svc.score(X_train, y_train), 4))
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        self.classifier.classifier = svc
        return self.classifier

    def add_training_file(self, file, label):
        self.files_list.append(file)
        self.labels_list.append(label)
