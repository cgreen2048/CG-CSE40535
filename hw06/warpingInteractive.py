# Special Studies: Computer Vision (CSE 40535)
# University of Notre Dame
# ______________________________________________________________________
# Adam Czajka, Toan Q. Nguyen, Siamul Khan, Walter Scheirer 2016 -- 2025

import numpy as np
import math
import cv2
from ROIPoly import roiPoly

roi = roiPoly(sort=False)

# Get selected image from roi poly object
I = cv2.cvtColor(np.array(roi.origImage), cv2.COLOR_RGB2BGR)
rows, cols, channels = I.shape

# Get transformation matrix using points from roi
# Go from the chosen points of the image to the corners of the output image
src = np.float32(roi.points)
dst = np.float32([[0, 0], [cols-1, 0], [cols-1, rows-1], [0, rows-1]])
# dst = np.float32([[cols-1, 0], [0, 0], [0, rows-1], [cols-1, rows-1]]) # dst matrix if sort is enabled

H_mat = cv2.getPerspectiveTransform(src, dst)
H_inv = np.linalg.inv(H_mat)

# Having matrix H we may do our transformation for each pixel:
I_transformed = np.zeros(I.shape).astype(np.uint8)

count = 0
for y_dest in range(0, rows):
    for x_dest in range(0, cols):
        
        destPX = np.float32([[x_dest], [y_dest], [1]])

        # *** The following line requires modification if you want to implement the "inverse warping":
        #  destPX = H_mat @ sourcePX
        sourcePX = H_inv @ destPX
        x_src = sourcePX[0,0] / sourcePX[2,0]
        y_src = sourcePX[1,0] / sourcePX[2,0]
        i = int(np.floor(x_src))
        j = int(np.floor(y_src))
        a = x_src % 1
        b = y_src % 1

        # bilinear interpolation of pixels around the computed source pixel
        # this allows us to get the color of surrounding pixels to determine what colors the destination pixel should be
        # this is the second step of "where is this pixel located in the original image?"
        destPX = (1-a)*(1-b)*I[j, i] + a*(1-b)*I[j, i+1] + (1-a)*b*I[j+1, i] + a*b*I[j+1, i+1]
        # x_dest = destPX[0,0]
        # y_dest = destPX[1,0]

        # x_dest = int(destPX[0,0]/destPX[2,0])
        # y_dest = int(destPX[1,0]/destPX[2,0])

        if 0 <= j and j < rows-1 and 0 <= i and i < cols-1:
            count = count + 1

            # *** The following line requires modification if you want to implement the "inverse warping":
            I_transformed[y_dest, x_dest, :] = destPX

I_correct_xformed = cv2.warpPerspective(I, H_mat, (cols, rows), flags=cv2.INTER_NEAREST)

cv2.imshow('Warped Images (left is yours, right is the correct one from library implementation)', np.concatenate([I_transformed, I_correct_xformed], axis=1))
print('This version of warping calculated new values for ', 100*count/(rows*cols), '% of destination pixels.')
cv2.waitKey(0)
cv2.destroyAllWindows()