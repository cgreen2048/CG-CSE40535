import cv2 as cv
import numpy as np

# you can add two images together with cv.add()
# or simply by np operation res = img1 + img2
# img2 can be another img or a scalar value

x = np.uint8([250])
y = np.uint8([10])

# 250 + 10 = 260 => 255
print( cv.add(x,y) )

# 250 + 10 = 260 % 256 = 4
print(x+y)

# stick with OpenCV funcs since they provide better result

# We can also do image addition as discussed in class for *blending*
# recall: g(x) = (1-a)f_0(x) + af_1(x)
# where we vary a from 0-1 to transition one img to another

img1 = cv.imread('spotify.jpg')
img2 = cv.imread('tree.jpg')

# 0.7 weighted img1 + 0.3 weighted img2
dst = cv.addWeighted(img1, 0.7, img2, 0.3, 0)

cv.imshow('blended image', dst)
cv.waitKey(0)
cv.destroyAllWindows()


# bitwise operations
# AND, OR, NOT, XOR are useful in masking operations to extract certain parts of images
img1 = cv.imread('messi5.jpg')
img2 = cv.imread('spotify.jpg')

# place img2 on top left corner, so we make an ROI
rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols ]

# create a mask of logo and create its inverse mask also
img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
mask_inv = cv.bitwise_not(mask)

# black out the area of logo in ROI
# this just means we take ROI where the mask is white, removing that part of the img
img1_bg = cv.bitwise_and(roi, roi, mask=mask_inv)

# take only the region of logo from logo image
img2_fg = cv.bitwise_and(img2, img2, mask=mask)

# put logo in ROI and modify main img
# what we do is add the blacked out background in img1 with the fg of img2 so we get just the logo

dst = cv.add(img1_bg, img2_fg)
img1[0:rows, 0:cols ] = dst

cv.imshow('res', img1)
cv.waitKey(0)
cv.destroyAllWindows()
