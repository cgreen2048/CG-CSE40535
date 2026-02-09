import numpy as np
import cv2

cam = cv2.VideoCapture(0)

backgrounds = []
while True:
    retval, img = cam.read()

    if not retval:
        break

    res_scale = 0.5 # rescale the input image if it's too large
    img = cv2.resize(img, (0,0), fx = res_scale, fy = res_scale)

    cv2.imshow("Background", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        print("Capturing background")
        backgrounds.append(img.copy())
    if key & 0xFF == 27:
        if len(backgrounds) == 0:
            backgrounds.append(img)
        break

background = np.median(backgrounds, axis=0).astype(np.uint8)

cv2.destroyAllWindows()

while True:
    retval, img = cam.read()

    if not retval:
        break

    res_scale = 0.5 # rescale the input image if it's too large
    img = cv2.resize(img, (0,0), fx = res_scale, fy = res_scale)

    # HSV values
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lowerLightBlue = np.array([30, 35, 150])
    upperLightBlue = np.array([60, 80, 255])
    
    # For each every pixel, checks if every pixel in that channel is w/in range
    # If it is, mask at that pixel = 255, else 0 (basically 1 or 0)
    mask = cv2.inRange(hsv, lowerLightBlue, upperLightBlue)


    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel=kernel)
    mask_inv = cv2.bitwise_not(mask)

    cv2.imshow("Final mask", mask)


    # bg pixel if mask = 255, black otherwise
    extracted_background = cv2.bitwise_and(background, background, mask=mask)

    img = cv2.bitwise_and(img, img, mask=mask_inv)

    img = cv2.add(img, extracted_background)

    cv2.imshow("Masked image", img)

    key = cv2.waitKey(1)
    if key & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()

