from scipy.misc import imresize
from skimage import data, color


def scale(screen_buffer, width=None, height=None, gray=False):
    if gray:
        gray_buffer = color.rgb2gray(screen_buffer)
    if width is not None and height is not None:
        return imresize(gray_buffer, (height, width))
    return gray_buffer
