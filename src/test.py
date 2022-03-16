#!/usr/bin/env python3
from argparse import ArgumentParser
import argparse
from genericpath import isfile
import cv2 as cv
from datetime import datetime as dt
from numpy import ndarray
from threading import Thread
from typing import Tuple
from os import mkdir
from os.path import splitext, expanduser, isdir, join
from queue import Queue

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

# frame queued and in wait to be processed
frames = Queue(100)


def get_video_type(filename: str) -> cv.VideoWriter_fourcc:
    filename, ext = splitext(filename)
    if ext in VIDEO_TYPE:
        return VIDEO_TYPE[ext]
    return VIDEO_TYPE["avi"]  # default in case filename has no extension


def write_frame(out: cv.VideoWriter, frame: ndarray) -> None:
    cloned_frame = frame["frame"].copy()
    cloned_frame = cv.putText(
        cloned_frame,  # frame to write on
        frame["date_time"],  # displayed text
        (10, 40),  # position on frame
        cv.FONT_HERSHEY_PLAIN,  # font
        1,  # font size
        (255, 255, 255),  # font color
        2,  # stroke
    )
    out.write(cloned_frame)


def set_cap_props(cap: cv.VideoCapture, res: str, fps: int) -> None:
    width, height = STD_DIMENSIONS[res]
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv.CAP_PROP_FPS, fps)


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
    argparser.add_argument(
        "-f",
        "--fps",
        type=int,
        metavar="<fps>",
        help="sets che fps for capture and video write",
        choices=[5, 10, 30, 60],
    )
    return argparser.parse_args()


def motion_detected(prev_frame, frame):
    proc_frame = cv.absdiff(prev_frame, frame)
    proc_frame = cv.cvtColor(proc_frame, cv.COLOR_BGR2GRAY)
    proc_frame = cv.GaussianBlur(proc_frame, (21, 21), 0)
    proc_frame = cv.threshold(proc_frame, 30, 255, cv.THRESH_BINARY)[1]
    proc_frame = cv.dilate(proc_frame, None, iterations=3)
    contours, _ = cv.findContours(
        proc_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    return contours


class ImageGrabber(Thread):
    def __init__(self, video: str, resolution: str, fps: int) -> None:
        Thread.__init__(self)
        if video:
            if isfile(video):
                # video input
                self.cap = cv.VideoCapture(expanduser(video))
            else:
                print("No video found")
                exit()
        else:
            # camera input
            self.cap = cv.VideoCapture(0)
        # default resolution
        resolution = resolution or "720p"
        # default fps
        fps = fps or 10
        set_cap_props(cap=self.cap, res=resolution, fps=fps)

    def run(self) -> None:
        global frames
        while self.cap.isOpened():
            _, frame = self.cap.read()
            frames.put(
                {
                    "date_time": dt.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "frame": frame,
                }
            )


class Main(Thread):
    def __init__(self, quiet: bool, cap: cv.VideoCapture) -> None:
        Thread.__init__(self)
        self.rec_delay = 3
        video_dir = join(expanduser("~"), "video")
        # output directory
        if not isdir(video_dir):
            mkdir(video_dir)
        filename = join(
            video_dir, f"{dt.today().strftime('%Y-%m-%d_%H:%M:%S')}.mkv"
        )
        video_type = get_video_type(filename)
        dims = (
            int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)),
        )
        self.fps = cap.get(cv.CAP_PROP_FPS)
        self.out = cv.VideoWriter(filename, video_type, self.fps, dims)
        if not quiet:
            print(
                "Starting recording:\n"
                "Resolution: "
                f"{int(cap.get(cv.CAP_PROP_FRAME_WIDTH))}x"
                f"{int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))}\n"
                f"Frames per second: {int(self.fps)}"
            )
        self.prev_frame = None

    def run(self) -> None:
        global frames
        self.prev_frame = frames.get()
        self.frame = frames.get()
        while True:
            if motion_detected(
                prev_frame=self.prev_frame["frame"], frame=self.frame["frame"]
            ):
                # if motion is detected record for REC_DELAY seconds
                for _ in range(int(self.rec_delay * self.fps)):
                    write_frame(out=self.out, frame=self.frame)
                    self.prev_frame = self.frame
                    self.frame = frames.get()
            else:
                self.prev_frame = self.frame
                self.frame = frames.get()


def main() -> None:
    args = get_args()
    grabber = ImageGrabber(
        video=args.video, resolution=args.resolution, fps=args.fps
    )
    m = Main(quiet=args.quiet, cap=grabber.cap)
    grabber.start()
    m.start()
    grabber.join()
    m.join()


if __name__ == "__main__":
    main()
