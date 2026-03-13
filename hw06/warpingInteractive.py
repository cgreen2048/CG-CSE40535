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
src = np.float32(roi.points)
dst = np.float32([[0, 0], [cols-1, 0], [cols-1, rows-1], [0, rows-1]])
# dst = np.float32([[cols-1, 0], [0, 0], [0, rows-1], [cols-1, rows-1]]) # dst matrix if sort is enabled

H_mat = cv2.getPerspectiveTransform(src, dst)

# Having matrix H we may do our transformation for each pixel:
I_transformed = np.zeros(I.shape).astype(np.uint8)

count = 0
for y_source in range(0, rows):
    for x_source in range(0, cols):
        
        sourcePX = np.float32([[x_source], [y_source], [1]])

        # *** The following line requires modification if you want to implement the "inverse warping":
        destPX = H_mat @ sourcePX
        
        x_dest = int(destPX[0,0]/destPX[2,0])
        y_dest = int(destPX[1,0]/destPX[2,0])

        if x_dest > 0 and y_dest > 0 and x_dest < cols and y_dest < rows:
            count = count + 1

            # *** The following line requires modification if you want to implement the "inverse warping":
            I_transformed[y_dest, x_dest, :] = I[y_source, x_source, :]

I_correct_xformed = cv2.warpPerspective(I, H_mat, (cols, rows), flags=cv2.INTER_NEAREST)

cv2.imshow('Warped Images (left is yours, right is the correct one from library implementation)', np.concatenate([I_transformed, I_correct_xformed], axis=1))
print('This version of warping calculated new values for ', 100*count/(rows*cols), '% of destination pixels.')
cv2.waitKey(0)
cv2.destroyAllWindows()