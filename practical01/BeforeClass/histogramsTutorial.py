import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# histograms are graphs/plots that give overall idea about
# the intensity distribution of an image
# plot with pixel vals (0...255 usually) in X-axis, corresponding # of pixels on Y axis


# BINS: sub-parts of the histogram where groups of pixels lie
# 0-15, 16-31...240-255 for 16 bins
# histSize in openCV

# DIMS: the number of params for which we collect the data
# in this case, we collect regarding one thing: intensity value
# 1 for this case

# RANGE: range of itensity values you want to measure
# Usually [0,256] which is all possible intensity vals

# use cv.calcHist() to find histogram
img = cv.imread('messi5.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read"
hist = cv.calcHist(
    [img],               # list of images
    [0],                 # index of channel for which we calc histogram. 0..2 for color
    None,                # mask image. None here so we can see hist of full img
    [256],               # histSize/Bins
    [0, 256]             # ranges   
)
# now, hist is a 256x1 array where each val corresponds to # of pixels in img with corresponding pixel value

# numpy version
npHist, bins = np.histogram(img.ravel(), 256, [0, 256])

# plotting histogram
# short way: matplotlib
plt.hist(img.ravel(), 256, [0,256]);
plt.show();