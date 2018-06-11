import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import random

import os
import glob
import helper
import matplotlib.image as mpimg

filenames = glob.glob('test_images/project_video_*.jpg')
classify_img_size = (64, 64)
neg_range_float = ((0, 0.35), (0.5, 1))
neg_samples = [
    (64, 16),
    (96, 12),
    (128, 4),
    (160, 2),
    ]

i = 0
for fname in filenames:
    ext = os.path.splitext(fname)[-1]
    if ext in ['.jpg', '.png']:
        img = helper.read_img(fname)
        width = img.shape[1]
        height = img.shape[0]
        neg_range = (
            (int(neg_range_float[0][0] * width), int(neg_range_float[0][1] * height)),
            (int(neg_range_float[1][0] * width), int(neg_range_float[1][1] * height)))
        for neg_sample_size, samples_per_size_per_img in neg_samples:
            for r in range(samples_per_size_per_img):
                offset_x = random.randint(neg_range[0][0], neg_range[1][0] - neg_sample_size)
                offset_y = random.randint(neg_range[0][1], neg_range[1][1] - neg_sample_size)
                neg_sample = img[offset_y:offset_y + neg_sample_size,offset_x:offset_x + neg_sample_size]
                if neg_sample_size != classify_img_size[0] or neg_sample_size != classify_img_size[1]:
                    neg_sample = cv2.resize(neg_sample, classify_img_size)
                helper.write_img(neg_sample, 'neg_augmentation_img/project_video_{}.jpg'.format(i))
                i += 1
    else:
        print("Unknown file extension:", fname)

print('Augmented', i, 'images')
