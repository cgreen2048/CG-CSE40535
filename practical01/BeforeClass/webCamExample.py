import cv2

cam = cv2.VideoCapture(0)

print("s -- Snapshot")
print("esc - Quit Program")

while True:
    retval, img = cam.read()
    res_scale = 0.5                     # rescale input if too large
    img = cv2.resize(img, (0,0), fx=res_scale, fy=res_scale)

    cv2.imshow("Live WebCam", img)

    action = cv2.waitKey(1)

    if action==27: break                # esc to close

    elif action==ord('s'):               # capture
        cap_img = img

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(cap_img,
                    "Hello I'm Clair Green!",
                    (10, 50),                   # start position
                    font,                       # font
                    1.0,                        # size
                    (255, 255, 255),             # BGR color
                    2,                          # thickness
                    cv2.LINE_AA)                # line type
        cv2.line(cap_img,
                 (100,100),
                 (300,300),
                 (0,0,0),
                 4)

        cv2.imshow("Captured iamge", cap_img)

