#!/usr/bin/env python3
from argparse import ArgumentParser
import argparse
from genericpath import isfile
import cv2 as cv
from datetime import datetime as dt
from numpy import ndarray
from typing import Tuple
from os import mkdir
from os.path import splitext, expanduser, isdir, join

VIDEO_DIR = join(expanduser("~"), "video")
REC_DELAY = 3
# Standard Video Dimensions Sizes
STD_DIMENSIONS = {
    "360p": (480, 360),
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
    "mkv": cv.VideoWriter_fourcc(*"XVID"),
    # 'mp4': cv2.VideoWriter_fourcc(*'H264'),
}


# Set resolution for the video capture
# Function adapted from https://kirr.co/0l6qmh
def change_res(cap: cv.VideoCapture, width: int, height: int) -> None:
    cap.set(3, width)
    cap.set(4, height)


# grab resolution dimensions and set video capture to it.
def get_dims(cap: cv.VideoCapture, res: Tuple[int, int]) -> Tuple[int, int]:
    width, height = STD_DIMENSIONS[res]
    if res[0] != cap.get(cv.CAP_PROP_FRAME_WIDTH) or res[1] != cap.get(
        cv.CAP_PROP_FRAME_HEIGHT
    ):
        # change the current caputre device
        # to the resulting resolution
        change_res(cap, width, height)
    return width, height


def get_video_type(filename: str) -> cv.VideoWriter_fourcc:
    filename, ext = splitext(filename)
    if ext in VIDEO_TYPE:
        return VIDEO_TYPE[ext]
    return VIDEO_TYPE["avi"]  # default in case filename has no extension


def write_frame(out: cv.VideoWriter, frame: ndarray) -> None:
    cloned_frame = frame.copy()
    cloned_frame = cv.putText(
        cloned_frame,  # frame to write on
        dt.now().strftime("%Y-%m-%d %H:%M:%S"),  # displayed text
        (10, 40),  # position on frame
        cv.FONT_HERSHEY_SIMPLEX,  # font
        1,  # font size
        (255, 255, 255),  # font color
        2,  # stroke
    )
    out.write(cloned_frame)


def set_cap_props(cap: cv.VideoCapture, dims: Tuple[int, int]) -> None:
    cap.set(cv.CAP_PROP_FRAME_WIDTH, dims[0])
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, dims[1])


def get_args() -> argparse.Namespace:
    argparser = ArgumentParser(allow_abbrev=False)
    argparser.add_argument(
        "-v",
        "--video",
        type=str,
        metavar="<path_to_vid>",
        help="path of the video file",
    )
    argparser.add_argument(
        "-r",
        "--resolution",
        type=str,
        metavar="<res>",
        help="resolution of the video capture",
        choices=["360p", "480p", "720p", "1080p", "4k"],
    )
    argparser.add_argument(
        "-q", "--quiet", action="store_true", help="wheter to mute the output"
    )
    return argparser.parse_args()


def motion_detected(prev_frame, frame) -> bool:
    proc_frame = cv.absdiff(prev_frame, frame)
    proc_frame = cv.cvtColor(proc_frame, cv.COLOR_BGR2GRAY)
    proc_frame = cv.GaussianBlur(proc_frame, (21, 21), 0)
    proc_frame = cv.threshold(proc_frame, 30, 255, cv.THRESH_BINARY)[1]
    proc_frame = cv.dilate(proc_frame, None, iterations=3)
    contours, _ = cv.findContours(
        proc_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    if len(contours) > 0:
        return True
    return False


def main() -> None:
    args = get_args()

    if args.video:
        if isfile(args.video):
            # video input
            cap = cv.VideoCapture(expanduser(args.video))
        else:
            print("No video found")
            exit()
    else:
        # camera input
        cap = cv.VideoCapture(0)

    # output directory
    if not isdir(VIDEO_DIR):
        mkdir(VIDEO_DIR)

    if not args.resolution:
        dims = get_dims(cap=cap, res="720p")
    else:
        dims = get_dims(cap=cap, res=args.resolution)
    filename = join(
        VIDEO_DIR, f"{dt.today().strftime('%Y-%m-%d_%H:%M:%S')}.mkv"
    )
    video_type = get_video_type(filename)
    set_cap_props(cap=cap, dims=dims)
    fps = cap.get(cv.CAP_PROP_FPS)

    out = cv.VideoWriter(filename, video_type, fps, dims)

    if not args.quiet:
        print(
            "Starting recording:\n"
            "Resolution: "
            f"{int(cap.get(cv.CAP_PROP_FRAME_WIDTH))}x"
            f"{int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))}\n"
            f"Frames per second: {int(cap.get(cv.CAP_PROP_FPS))}"
        )

    _, prev_frame = cap.read()
    _, frame = cap.read()
    while cap.isOpened():
        try:
            if motion_detected(prev_frame=prev_frame, frame=frame):
                # if motion is detected record for REC_DELAY seconds
                for _ in range(int(REC_DELAY * fps)):
                    write_frame(out=out, frame=frame)
                    prev_frame = frame
                    _, frame = cap.read()
            else:
                prev_frame = frame
                _, frame = cap.read()
        except KeyboardInterrupt:
            cap.release()
            out.release()
            break


if __name__ == "__main__":
    main()
