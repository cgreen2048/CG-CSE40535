import numpy as np
import cv2 as cv

img = cv.imread('messi5.jpg')
assert img is not None, "file could not be read"

px = img[100,100]
print(px)

# blue pixel
px = img[100,100,0]
print(px)

img[100,100] = [255,255,255]
print(img[100,100])

print(f"img shape: {img.shape}")
print(f"img # of pixels: {img.size}")


# extraction of certain regions of image
ball = img[280:340, 330:390]

# splitting image channels to work on b,g,r separately
b,g,r = cv.split(img)
img = cv.merge((b,g,r))

blue = img[:,:,0]

# set all red pixels to 0
# 1st index is w, 2nd h, 3rd channel
# channels are b,g,r in order
img[:,:,2] = 0

# display image in cv2 window
while True:
    cv.imshow("Img display", img);

