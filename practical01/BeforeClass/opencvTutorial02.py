import cv2
import numpy as np

# we will change between color spaces from BGR to HSV

# first, print every single color space flag just to see over 150 total
flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
# print(flags)

# BGR --> HSV
cap = cv2.VideoCapture(0)

# this is the simplest method in object tracking
# we simply just extract all blue which creates a border for blue elements
while True:
    _, frame = cap.read()

    # convert the frame from BGR to HSV color space
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of forest green in HSV
    # lower_blue = np.array([57,100,100])
    # upper_blue = np.array([77,255,255])

    # blue instead
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    # Threshold the HSV image to get only blue colors in our set range
    mask = cv2.inRange(hsvFrame, lower_blue, upper_blue)

    # bitwise AND for mask and original image to extract
    res = cv2.bitwise_and(frame,frame, mask=mask)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

# general strategy to extract an HSV color
# color = np.uint8([[[r,g,b]]])
# hsv_color = cv2.cvtColor(color, cv.COLOR_BGR2HSV)
# grab the color from hsv_color
# lower_color = [H-10, 100, 100]
# upper_color = [H+10, 255, 255]
# mask = cv2.inRange(img, lower_color, upper_color)

color = np.uint8([[[79, 121, 66]]])
hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
print(hsv_color)
lower_color = [57, 100, 100]
upper_color = [77, 255, 255]
