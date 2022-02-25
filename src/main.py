#!/usr/bin/env python3
from typing import Tuple
import cv2 as cv
from datetime import datetime as dt
from time import sleep
from os.path import splitext

filename = "test.mp4"
frames_per_second = 10.0
res = "720p"

# Standard Video Dimensions Sizes
STD_DIMENSIONS = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}

# Video Encoding, might require additional installs
# Types of Codes: http://www.fourcc.org/codecs.php
VIDEO_TYPE = {
    "avi": cv.VideoWriter_fourcc(*"XVID"),
    "mp4": cv.VideoWriter_fourcc(*"XVID"),
    # 'mp4': cv2.VideoWriter_fourcc(*'H264'),
}


# Set resolution for the video capture
# Function adapted from https://kirr.co/0l6qmh
def change_res(cap: cv.VideoCapture, width: int, height: int) -> None:
    cap.set(3, width)
    cap.set(4, height)


# grab resolution dimensions and set video capture to it.
def get_dims(
    cap: cv.VideoCapture, res: Tuple[int, int] = "1080p"
) -> Tuple[int, int]:
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width, height = STD_DIMENSIONS[res]
    # change the current caputre device
    # to the resulting resolution
    change_res(cap, width, height)
    return width, height


def get_video_type(filename: str) -> cv.VideoWriter_fourcc:
    filename, ext = splitext(filename)
    if ext in VIDEO_TYPE:
        return VIDEO_TYPE[ext]
    return VIDEO_TYPE["avi"]  # default in case filename has no extension


def main() -> None:
    cap = cv.VideoCapture(0)
    dims = get_dims(cap, res)
    video_type = get_video_type(filename)
    out = cv.VideoWriter(filename, video_type, frames_per_second, dims)

    motion = False
    recording = False
    _, prev_frame = cap.read()
    _, frame = cap.read()

    while cap.isOpened():
        try:
            proc_frame = cv.absdiff(prev_frame, frame)
            proc_frame = cv.cvtColor(proc_frame, cv.COLOR_BGR2GRAY)
            proc_frame = cv.GaussianBlur(proc_frame, (21, 21), 0)
            proc_frame = cv.threshold(proc_frame, 30, 255, cv.THRESH_BINARY)[1]
            proc_frame = cv.dilate(proc_frame, None, iterations=3)
            contours, _ = cv.findContours(
                proc_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )
            # if contours are detected, then motion occours
            if len(contours) > 0:
                out.write(frame)
                motion = True
            # if motion != recording:
            #     # motion detected but recording is not running, so start
            #     # recording
            #     if motion is True:
            #         print("Rec started")
            #         recording = True
            #     # motion not detected but recording is running, so stop
            #     # recording
            #     else:
            #         print("Rec stopped")
            #         recording = False
            # motion = False

            # prepare for the next cycle
            prev_frame = frame
            _, frame = cap.read()
        except KeyboardInterrupt:
            cap.release()
            out.release()
            break


if __name__ == "__main__":
    main()
