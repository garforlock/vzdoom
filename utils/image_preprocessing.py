import skimage
import numpy as np
import cv2

from skimage import data, color


def scale(screen_buffer, x=None, y=None, gray=False):

    gray_buffer = screen_buffer
    if gray:
        gray_buffer = color.rgb2gray(screen_buffer)
    if x is not None and y is not None:
        # return imresize(gray_buffer, (x, y))
        return cv2.resize(gray_buffer, dsize=(x, y))

    return gray_buffer
# Converts and down-samples the input image


def preprocess(img, resolution=(30, 45)):
    img = skimage.transform.resize(img, resolution, mode='constant')
    img = img.astype(np.float32)
    return img
