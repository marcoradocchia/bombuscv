#!/usr/bin/env python3
import cv2 as cv
import numpy as np
from sys import exit

# WARNING: warning for missing gstreamer dependency


def main() -> None:
    cap = cv.VideoCapture(0)
    if not cap.isOpened():  # check if camera is open
        exit("Cannot open camera")
    while True:
        # capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True, else exit
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        orig_frame = frame.copy()
        # convert frame to gray scale
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = cv.GaussianBlur(frame, (21, 21), cv.BORDER_DEFAULT)
        circles = cv.HoughCircles(
            frame,
            cv.HOUGH_GRADIENT_ALT,
            0.9,
            50,
            param1=100,
            param2=0.9,
            minRadius=20,
            maxRadius=0,
        )
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # draw circles on frame
            for i in circles[0, :]:
                cv.circle(orig_frame, (i[0], i[1]), i[2], (50, 100, 255), 2)

        # display the resulting frame
        cv.imshow("frame", orig_frame)
        # quit on q pressed
        if cv.waitKey(50) == ord("q"):
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
